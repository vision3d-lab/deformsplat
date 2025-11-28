# This file includes code from https://github.com/jkxing/DROT/blob/main/core/LossFunction.py,
# which is licensed under the MIT License.

# MIT License

# Copyright (c) 2022 DROT

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from torch.nn import Module



class PointLossFunction(Module):
    def __init__(
        self,
        resolution=(100,100),
        renderer=None,
        device="cuda",
        settings={},
        debug=False,
        num_views=1,
        logger=None,
    ):
        super().__init__()
        self.num_views = num_views

        self.match_weight = settings.get("matching_weight", 1.0)
        self.matchings_count = [0 for i in range(num_views)]
        self.matchings = [[] for i in range(num_views)]
        self.rasts = [[] for i in range(num_views)]
        self.rgb_weight = [self.match_weight for i in range(num_views)]
        self.matching_interval = settings.get("matching_interval", 0)
        self.renderer = renderer
        self.device = device
        self.h = resolution[0]
        self.w = resolution[1]
        self.resolution = max(self.h, self.w)
        self.debug = debug
        self.logger = logger
        self.step = -1
        # Matcher setting
        self.matcher_type = settings.get("matcher", "Sinkhorn")
        self.matcher = None
        from geomloss import SamplesLoss
        self.loss = SamplesLoss("sinkhorn", blur=0.01)

        # normal image grid, used for pixel position completion
        x = torch.arange(0, self.w) / max(self.h, self.w)
        y = torch.arange(0, self.h) / max(self.h, self.w)
        y_grid, x_grid = torch.meshgrid(x, y, indexing="xy")  # (2, r, r)
        self.uv = (
            torch.stack([y_grid, x_grid], dim=-1)
            .to(self.device)[None, ...]
            .repeat(num_views, 1, 1, 1)
        )
        self.uv_np = self.uv[0].clone().cpu().numpy().reshape(-1, 2)  # (r*r, 2)

    def visualize_point(self, res, title, view):  # (N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        X = res[..., 3:]
        # need install sklearn
        nbrs = None  # NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.uv_np)
        distances = np.exp(-distances * self.resolution)
        img = np.sum(res[indices, :3] * distances[..., None], axis=1)
        img = img / np.sum(distances, axis=1)[..., None]
        img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title + "_" + str(view), img, self.step)

    # unused currently
    def rgb_match_weight(self, view=0):
        return self.rgb_weight[view]

    def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
        h, w = render_point_5d.shape[1:3]
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb, 0, 1)
        target_point_5d[..., 3:] = render_point_5d[..., 3:].clone().detach()
        target_point_5d = target_point_5d.reshape(-1, h * w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1, h * w, 5)
        render_point_5d_match.clamp_(0.0, 1.0)
        render_point_5d_match[..., :3] *= self.rgb_match_weight(view)
        target_point_5d[..., :3] = target_point_5d[..., :3] * self.rgb_match_weight(
            view
        )
        pointloss = (
            self.loss(render_point_5d_match, target_point_5d)
            * self.resolution
            * self.resolution
        )
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        g[..., :3] /= self.rgb_match_weight(view)
        return (render_point_5d - g.reshape(-1, h, w, 5)).detach()

    def get_loss(self, render_result, gt_rgb, view, return_matching):
        haspos = render_result["msk"]

        render_uv = (render_result["pos"] + 1.0) / 2.0
        render_rgb = render_result["images"][..., :3]

        # if there is no position, replace the position for non grad
        render_uv[haspos == False] = self.uv[view : view + 1][haspos == False].clone()
        render_point_5d = torch.cat([render_rgb, render_uv], dim=-1)
        match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)
        disp = match_point_5d - render_point_5d
        loss = torch.sum(disp**2)
        if return_matching:
            return loss, match_point_5d
        else:
            return loss

    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step = iteration

        new_match = (self.matchings_count[view] % (self.matching_interval + 1)) == 0

        if new_match:
            render_result = self.renderer.render(scene, view=view, DcDt=False)
            self.rasts[view] = render_result["rasts"]
        else:
            render_result = self.renderer.render(
                scene, rasts_list=self.rasts[view], view=view, DcDt=False
            )

        self.matchings_count[view] += 1
        haspos = render_result["msk"]
        render_uv = (render_result["pos"] + 1.0) / 2.0
        render_rgb = render_result["images"]
        render_uv[haspos == False] = self.uv[view : view + 1][haspos == False].clone()
        render_point_5d = torch.cat([render_rgb, render_uv], dim=-1)
        gt_rgb = gt["images"][view : view + 1]
        if new_match:
            if self.matcher_type == "Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(
                    haspos, render_point_5d, gt_rgb, view
                )

        match_point_5d = self.matchings[view]
        disp = match_point_5d - render_point_5d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(
                match_point_5d.reshape(-1, 5), title="match", view=view
            )

        return loss, render_result
