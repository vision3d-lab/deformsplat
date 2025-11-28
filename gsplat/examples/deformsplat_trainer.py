import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from simple_trainer import Runner
from simple_trainer import *

from util.loss import arap_loss, drag_loss, arap_loss_grouped
from util.mini_pytorch3d import quaternion_to_matrix
from jhutil.algorithm import knn as knn_jh
from util.mini_pytorch3d import quaternion_multiply
from util.roma import get_drag_roma
from util.rigid_grouping import local_rigid_grouping, naive_rigid_grouping, refine_rigid_group
from util.helper import (
    rbf_weight,
    voxelize_pointcloud_and_get_means,
    linear_blend_skinning_knn,
    cluster_largest,
    project_pointcloud_to_2d,
    count_covered_patches,
)

from jhutil import show_matching, color_log
import wandb


class DeformRunner(Runner):
    def train_deformsplat(
        self,
        drag_iterations=500,
    ) -> None:
        if not self.cfg.skip_eval:
            step = 0
            self.eval(step=step)

        # get haraparameter
        coef_drag            = self.hpara.coef_drag
        coef_arap_drag       = self.hpara.coef_arap_drag
        coef_group_arap      = 0 if self.cfg.without_group else self.hpara.coef_group_arap
        coef_rgb             = self.hpara.coef_rgb
        coef_drag            = self.hpara.coef_drag
        lr_q                 = self.hpara.lr_q
        lr_t                 = self.hpara.lr_t
        rigidity_k           = self.hpara.rigidity_k
        reprojection_error   = self.hpara.reprojection_error
        anchor_k             = self.hpara.anchor_k
        rbf_gamma            = self.hpara.rbf_gamma
        cycle_threshold      = self.hpara.cycle_threshold
        min_inlier_ratio     = self.hpara.min_inlier_ratio
        confidence           = self.hpara.confidence
        refine_radius        = self.hpara.refine_radius
        refine_threhold      = self.hpara.refine_threhold
        voxel_size           = self.hpara.voxel_size
        filter_distance      = self.hpara.filter_distance
        min_group_size       = self.hpara.min_group_size
        
        self.splats = dict(self.splats)
        points_init = self.splats["means"].clone().detach()
        quats_init = self.splats["quats"].clone().detach()

        ##########################################################
        ############ 1. get campose and drag via RoMa ############
        ##########################################################

        color_log(1111, "get campose and drag via RoMa")
        with torch.no_grad():

            self.image_target = self.fetch_target_image(return_rgba=True)
            self.camtoworlds_pred = self.estimate_camtoworlds(self.image_target)

            image_source, image_target = self.fetch_comparable_two_image(
                return_rgba=True
            )

            drag_source, drag_target, bbox = get_drag_roma(
                image_source, image_target, device=self.device, cycle_threshold=cycle_threshold
            )
            self.height = image_source.shape[1]
            self.width = image_source.shape[2]

        ##########################################################
        ############### 2. filter points and drag  ###############
        ##########################################################
        color_log(2222, "filter points and start drag")
        points_3d = self.splats["means"].clone().detach()
        points_3d.requires_grad = True

        # set drag
        with torch.no_grad():
            points_2d, points_depth = self.project_to_2d(points_3d)

        vis_mask = self.get_visibility_mask()
        from util.helper import get_drag_mask
        points_mask, drag_indice = get_drag_mask(
            points_2d, vis_mask, drag_source, filter_distance
        )

        points_3d_filtered = points_3d[points_mask]
        drag_target_filtered = drag_target[drag_indice]
        drag_source_filtered = drag_source[drag_indice]

        ##########################################################
        ########### 3. initialize anchor and optimizer ###########
        ##########################################################
        color_log(3333, "initialize anchor and optimizer")

        anchor = voxelize_pointcloud_and_get_means(points_3d, voxel_size=voxel_size)
        
        anchor = anchor.to(self.device)


        N = anchor.shape[0]
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        quats = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)

        anchor_optimizer = torch.optim.Adam(
            [
                {"params": quats, "lr": lr_q},
                {"params": t, "lr": lr_t},
            ]
        )

        ##########################################################
        #################### 4. rigid grouping ###################
        ##########################################################
        color_log(4444, "rigid grouping")

        if self.cfg.naive_group:
            groups = naive_rigid_grouping(
                points_3d_filtered,
                drag_target_filtered,
                reprojection_error,
                camera_matrix=self.data["K"][0]
            )
        else:
            groups, outliers, group_trans = local_rigid_grouping(
                points_3d_filtered,
                drag_target_filtered,
                k=rigidity_k,
                min_inlier_ratio=min_inlier_ratio,
                confidence=confidence,
                min_group_size=min_group_size,
                max_expansion_iterations=100,
                reprojection_error=reprojection_error,
                iterations_count=100,
                camera_matrix=self.data["K"][0],
            )

        groud_id = -torch.ones(
            points_3d_filtered.shape[0], dtype=torch.long, device=self.device
        )
        for i, group in enumerate(groups):
            groud_id[group] = i
        group_id_all = -torch.ones(
            points_3d.shape[0], dtype=torch.long, device=self.device
        )
        group_id_all[points_mask] = groud_id
        group_id_all_init = group_id_all.clone().detach()
        if wandb.run and not self.cfg.wandb_sweep:
            n_drag = len(drag_source)
            n_pts = 5000

            img1 = rearrange(image_source[0], "h w c -> c h w")
            img2 = rearrange(image_target[0], "h w c -> c h w")
            
            origin_image = show_matching(img1, img2, bbox=bbox, skip_line=True)
            matching_image = show_matching(
                img1,
                img2,
                drag_source[:: n_drag // n_pts],
                drag_target[:: n_drag // n_pts],
                bbox=bbox,
                skip_line=True,
            )
            for i in range(0, 5):
                group_id_all_tmp = torch.where(
                    group_id_all == i, group_id_all, -1
                )
                sh0_origin = self.update_sh_with_group_id(group_id_all_tmp)
                with torch.no_grad():
                    group_image, _ = self.fetch_comparable_two_image(return_rgba=True)

                group_image = rearrange(group_image[0], "h w c -> c h w")
                from jhutil import show_matching as _show_matching
                group_image = _show_matching(img1, group_image, bbox=bbox, skip_line=True)
                self.splats["sh0"] = sh0_origin
                
                images = [
                    wandb.Image(origin_image, caption="origin_image"),
                    wandb.Image(matching_image, caption="matching_image"),
                ]
                wandb.log({"matching": images})
                wandb.log({"group_image": [wandb.Image(group_image, caption="initial_group")]})
        

        ##########################################################
        ############## 5. DeformSplat optimization ###############
        ##########################################################
        color_log(5555, "DeformSplat optimization ")
        
        quats_origin = F.normalize(self.splats["quats"].clone().detach(), dim=-1)
        
        with torch.no_grad():
            distances, indices_knn = knn_jh(anchor, anchor, k=anchor_k)
            weight = rbf_weight(distances, gamma=rbf_gamma)

        for i in tqdm(range(drag_iterations + 1)):
            
            R = quaternion_to_matrix(F.normalize(quats, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)

            points_lbs, quats_lbs = linear_blend_skinning_knn(points_3d, anchor, R, t)
            updated_quaternions = quaternion_multiply(quats_lbs, quats_origin)

            R_points = quaternion_to_matrix(F.normalize(quats_lbs, dim=-1)).squeeze()
            loss_group_arap = arap_loss_grouped(
                points_3d, points_lbs, R_points, group_id_all
            )

            points_lbs_filtered = points_lbs[points_mask]
            points_lbs_filtered_2d, _ = self.project_to_2d(points_lbs_filtered)
            loss_drag = drag_loss(points_lbs_filtered_2d, drag_target_filtered)

            self.splats["means"] = points_lbs
            self.splats["quats"] = updated_quaternions
            loss_rgb = self.render_and_calc_rgb_loss() if i > 300 else 0

            loss = (
                coef_drag * loss_drag
                + coef_arap_drag * loss_arap
                + coef_group_arap * loss_group_arap
                + coef_rgb * loss_rgb
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            anchor_optimizer.zero_grad()

            if not self.cfg.without_group_refine and i > 300 and i % 10 == 0:
                group_id_all = refine_rigid_group(
                    points_3d,
                    points_lbs,
                    group_id_all,
                    R_points,
                    radius=refine_radius,
                    outlier_threhold=refine_threhold,
                )
                
            if wandb.run and not self.cfg.wandb_sweep:
                
                if i % 10 == 0:
                    wandb.log({
                        "loss_arap"      : loss_arap,
                        "loss_drag"      : loss_drag,
                        "loss_group_arap": loss_group_arap,
                        "loss_rgb"       : loss_rgb
                    }, step=i)

                if i % 100 == 0:
                    image_source, image_target = self.fetch_comparable_two_image(
                        return_rgba=True, return_shape="chw"
                    )
                    from jhutil import crop_two_image_with_alpha, get_img_diff
                    _, img1, img2 = crop_two_image_with_alpha(
                        image_source, image_target
                    )
                    diff_img = get_img_diff(img1, img2)
                    
                    wandb.log(
                        {"train_diff": [
                            wandb.Image(diff_img, caption="train_diff"),
                        ]}, step=i,
                    )

        if not self.cfg.skip_eval:
            self.eval(step=drag_iterations+1, is_pred_camtoworld=True)


        sh0_origin = self.update_sh_with_group_id(group_id_all)
        self.log_final_group(points_init, quats_init, bbox)
        
        self.splats["sh0"] = sh0_origin

        # data for motion
        motion_data = [points_init.detach(), quats_init.detach(), anchor.detach(), R.detach(), t.detach(), group_id_all, bbox, self.camtoworlds_pred]
        torch.save(motion_data, f"{self.cfg.result_dir}/motion_data.pt")

        camtoworlds_origin = self.data["camtoworld"].to(self.device)
        camtoworlds_residual = self.camtoworlds_pred @ camtoworlds_origin.inverse()
        
        # point [x, y, z] to [x, y, z, 1]
        means_homo = torch.concat([self.splats["means"], torch.ones((self.splats["means"].shape[0], 1), device=self.device)], dim=1)
        self.splats["means"] = means_homo @ camtoworlds_residual[0][:, :3]
        
        from util.mini_pytorch3d import matrix_to_quaternion
        quat_residual = matrix_to_quaternion(camtoworlds_residual[0, :3, :3].T)
        self.splats["quats"] = quaternion_multiply(quat_residual, self.splats["quats"])
        
        # save checkpoint
        data = {"step": step, "splats": (torch.nn.ParameterDict(self.splats).state_dict()), "group_id_all": group_id_all, "group_id_all_init": group_id_all_init}
        torch.save(data, f"{self.cfg.result_dir}/ckpt_finetune.pt")
        print("save checkpoint to ", f"{self.cfg.result_dir}/ckpt_finetune.pt")


    def log_final_group(self, points_init, quats_init, bbox):
        with torch.no_grad():
            group_image, image_target = self.fetch_comparable_two_image(return_rgba=True)

        im1 = rearrange(image_target[0], "h w c -> c h w")
        im2 = rearrange(group_image[0], "h w c -> c h w")

        image_final_group = show_matching(im1, im2, bbox=bbox, skip_line=True)
        wandb.log({"group_image": [wandb.Image(image_final_group, caption="final_group")]})
        
        points_final = self.splats["means"].clone().detach()
        quats_final = self.splats["quats"].clone().detach()

        with torch.no_grad():
            self.splats["means"] = points_init
            self.splats["quats"] = quats_init
            group_image, image_target = self.fetch_comparable_two_image(return_rgba=True)
            self.splats["means"] = points_final
            self.splats["quats"] = quats_final
            im2 = rearrange(group_image[0], "h w c -> c h w")
        wandb.log({"group_image": [wandb.Image(im2, caption="final_group_2")]})


    def get_visibility_mask(self):

        means3d = self.splats["means"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        scales = self.splats["scales"]
        Ks = self.data["K"].to(self.device)
        camtoworlds = self.data["camtoworld"].to(self.device)
        visibility, means2d = compute_visibility(self.width, self.height, means3d, quats, opacities, scales, Ks, camtoworlds)
        visible_mask = visibility > self.hpara.vis_threshold

        return visible_mask


    def make_motion_video(self, idx=0, n_iter=1000, threhold_early_stop=1e-5 , scheduler_step=300, min_rigid_coef=0):
        color_log(6666, "postprocess for smooth deformation")
        motion_data = torch.load(f"{self.cfg.result_dir}/motion_data.pt")

        # TODO: delete
        try:
            points_init, quats_init, anchor, R_goal, t_goal, group_id_all, bbox, self.camtoworlds_pred = motion_data
        except:
            points_init, quats_init, anchor, R_goal, t_goal, group_id_all, bbox = motion_data
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
            self.camtoworlds_pred = self.data["camtoworld"].to(self.device)
    

        self.splats["means"].data.copy_(points_init.detach())
        self.splats["quats"].data.copy_(quats_init.detach())
        
        # get haraparameter
        coef_drag            = self.hpara.coef_drag_3d
        coef_arap_drag       = self.hpara.coef_arap_drag
        if self.cfg.data_name == "dfa":
            coef_arap_drag *= 0.1

        lr                   = self.hpara.lr_motion
        anchor_k             = self.hpara.anchor_k
        rbf_gamma            = self.hpara.rbf_gamma
        
        
        ##########################################################
        ########### a. initialize anchor and optimizer ###########
        ##########################################################
        
        quats_origin = F.normalize(self.splats["quats"].clone().detach(), dim=-1)
        points_3d = self.splats["means"].clone().detach()
        
        self.splats["quats"].data.copy_(quats_origin)
        

        anchor = anchor.to(self.device)
        N = anchor.shape[0]
        # re-init parameter
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        q = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)

        anchor_optimizer = torch.optim.Adam([q, t], lr=lr, weight_decay=0.01)
        scheduler = LambdaLR(anchor_optimizer, lr_lambda=lambda step: 0.3 + min(0.7, (step + 1) / scheduler_step))
        ##########################################################
        #################### b. drag optimize ####################
        ##########################################################
        if not self.cfg.disable_viewer:
            breakpoint()
        
        with torch.no_grad():
            distances, indices_knn = knn_jh(anchor, anchor, k=anchor_k)
            weight = rbf_weight(distances, gamma=rbf_gamma)

        loss_fn = SmoothL1Loss(beta=0.1)
        
        # change into until convergence
        image_list = []
        for i in tqdm(range(n_iter)):
            R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)
            loss_drag = loss_fn(R, R_goal) + loss_fn(t, t_goal)
            
            if loss_drag < threhold_early_stop:
                break

            points_lbs, quats_lbs = linear_blend_skinning_knn(points_3d, anchor, R, t)
            updated_quaternions = quaternion_multiply(quats_lbs, quats_origin)

            R_points = quaternion_to_matrix(F.normalize(quats_lbs, dim=-1)).squeeze()
            loss_group_arap = arap_loss_grouped(
                points_3d, points_lbs, R_points, group_id_all
            )

            self.splats["means"].data.copy_(points_lbs)
            self.splats["quats"].data.copy_(updated_quaternions)
            
            loss = (
                coef_drag * loss_drag
                # + max(min_rigid_coef, 1 - i / scheduler_step) * coef_group_arap * loss_group_arap
                + max(min_rigid_coef, 1 - i / scheduler_step) * coef_arap_drag * loss_arap
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            scheduler.step()
            anchor_optimizer.zero_grad()

            with torch.no_grad():
                image_source, image_target = self.fetch_comparable_two_image(return_rgba=True)
            
            if self.cfg.simple_video:
                w_from, h_from, w_to, h_to = bbox
                image = image_source[:, h_from-5:h_to+5, w_from-10:w_to].squeeze()
            else:
                # crop
                # w_from, h_from, w_to, h_to = bbox
                # image_source = image_source[:, h_from-30:h_to+30, w_from-45:w_to+15]
                # image_target = image_target[:, h_from-30:h_to+30, w_from-45:w_to+15]
                image_target[..., :3] = image_target[..., :3] + (1 - image_target[..., 3:])
                # src and tgt
                image = torch.cat(
                    [image_source.squeeze(), image_target.squeeze()], dim=1
                )
            image_list.append(image.cpu())


        stride = max(len(image_list), 75) // 75
        if self.cfg.simple_video:
            output_path = self.cfg.video_path.replace(".mp4", "_deformation.mp4")
            
            image_list = torch.stack(image_list, dim=0)
            
            image_list_path = f"./output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}.pt"
            torch.save(image_list, image_list_path)

            img_path = f"./output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            indice = [0, int(len(image_list) * 0.2), int(len(image_list) * 0.35), -1]
            save_motion_img(image_list[indice], img_path)

            img_path = f"./output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}_low_alpha.png"
            indice = [0, int(len(image_list) * 0.2), -1]
            save_motion_img(image_list[indice], img_path, alpha_range=(0.2, 1))


        else:
            output_path = self.cfg.video_path.replace(".mp4", "_deformation.mp4")
            # rewind
            image_list = image_list + [image_list[-1]] * 30 + image_list[::-1]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(image_list[::stride], output_path, fps=30)


    def project_to_2d(self, points, use_gt_pose=False):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
        data = self.data
        device = self.device
        # camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]

        if use_gt_pose:
            camtoworlds = data["camtoworld"].to(device)
        else:
            camtoworlds = self.camtoworlds_pred
        
        Ks = data["K"].to(device)  # [1, 3, 3]
        means2d, depth = project_pointcloud_to_2d(points, camtoworlds, Ks)
        return means2d, depth


    def fectch_query_image(self):
        device = self.device
        
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)
        gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        gt_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
        
        gt_images = torch.concat([gt_images, gt_alphas], dim=-1)
        
        return gt_images
    

    def estimate_camtoworlds(self, image_target):
        trainset = Dataset(
            self.parser,
            split="train",
            patch_size=self.cfg.patch_size,
            load_depths=self.cfg.depth_loss,
            single_finetune=False,
            cam_idx=self.cfg.cam_idx,
        )
        trainloader = DataLoader(trainset, batch_size=1)
        device = self.device

        max_covered_path = -1
        for data in trainloader:
            Ks = data["K"].to(device)  # [1, 3, 3]
            image_ids = data["image_id"].to(device)
            gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = gt_images.shape[1:3]
            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=3,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
            )
            image_source = torch.concat([renders, alphas], dim=-1)
            
            img1 = rearrange(image_source[0], "h w c -> c h w")
            img2 = rearrange(image_target[0], "h w c -> c h w")

            assert img1.shape == img2.shape, f"img1: {img1.shape}, img2: {img2.shape}"
            try:
                drag_from, drag_to, bbox = get_drag_roma(img1, img2, cycle_threshold=self.hpara.cycle_threshold, device=device)
            except:
                try:
                    drag_from, drag_to, bbox = get_drag_roma(img1, img2, cycle_threshold=self.hpara.cycle_threshold, device=device)
                except:
                    continue
                    
            n_covered_patch = count_covered_patches(drag_to, patch_size=20)
            if n_covered_patch > max_covered_path:
                max_covered_path = n_covered_patch
                camtoworld_pred = camtoworlds

        return camtoworld_pred


    def fetch_target_image(self, return_rgba=False, return_shape="bhwc"):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
        data = self.data
        device = self.device

        target_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        target_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]

        if return_rgba:
            target_images = torch.concat([target_images, target_alphas], dim=-1)
        if return_shape == "bhwc":
            pass
        elif return_shape == "chw":
            target_images = target_images[0].permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid shape: {return_shape}")

        return target_images

    
    def fetch_comparable_two_image(self, return_rgba=False, return_shape="bhwc", use_gt_pose=False):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)

        data = self.data
        device = self.device

        Ks = data["K"].to(device)  # [1, 3, 3]
        gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        gt_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        height, width = gt_images.shape[1:3]

        if use_gt_pose:
            camtoworlds = data["camtoworld"].to(device)
        else:
            camtoworlds = self.camtoworlds_pred
        
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=3,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
            masks=masks,
        )

        if return_rgba:
            renders = torch.concat([renders, alphas], dim=-1)
            gt_images = torch.concat([gt_images, gt_alphas], dim=-1)

        if return_shape == "bhwc":
            pass
        elif return_shape == "chw":
            renders = renders[0].permute(2, 0, 1)
            gt_images = gt_images[0].permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid shape: {return_shape}")

        return renders, gt_images


    def render_and_calc_rgb_loss(self):
        renders, gt_images = self.fetch_comparable_two_image()
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None
        rgb_loss = F.l1_loss(colors, gt_images)

        return rgb_loss


    @torch.no_grad()
    def render_traj(self, step, group_id_all=None, video_path="./output_video_traj/output.mp4"):

        self.backgrounds = torch.ones(1, 3, device=self.device)   # white
        
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if group_id_all is not None:
            sh0_origin = self.update_sh_with_group_id(group_id_all)

        camtoworlds_all = self.parser.camtoworlds
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 4
            )  # [N, 3, 4]
            
        if cfg.render_traj_path == "dfa_interp":
            indices = list(range(cfg.cam_idx, -1, -1)) + list(range(len(camtoworlds_all) - 1, cfg.cam_idx - 1, -1))
            camtoworlds_all = camtoworlds_all[indices]
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 4
            )  # [N, 3, 4] + 
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        elif cfg.render_traj_path == "diva360_circle":
            object_name = cfg.object_name.split("_[")[0]
            json_path = f"./gsplat/data/Diva360_data/processed_data/{object_name}/transforms_circle.json"
            camtoworlds_all = json_to_cam2world(json_path, self.parser.transform)
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]

        elif cfg.render_traj_path == "diva360_spiral":
            object_name = cfg.object_name.split("_[")[0]
            json_path = f"./gsplat/data/Diva360_data/processed_data/{object_name}/transforms_spiral_hr.json"
            camtoworlds_all = json_to_cam2world(json_path, self.parser.transform)
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
            camtoworlds_all = camtoworlds_all[220:385]


        elif cfg.render_traj_path == "diva360_interp":
            # camtoworlds_all = transform_cameras(np.linalg.inv(self.parser.transform), camtoworlds_all)
            # camtoworlds_all = camtoworlds_all[[0,1,7,11,12,18,0]]
            camtoworlds_all = generate_interpolated_path_circle(
                camtoworlds_all, 100
            )  # [N, 3, 4]

        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # video_dir from video_path
        video_dir = os.path.dirname(video_path)
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=30)
        for i in tqdm(range(len(camtoworlds_all)), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=int(height * 1.2),
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            H, W = colors.shape[1:3]
            # colors = colors[:, :, W//4:3*W//4]  # crop by width

            canvas = colors.squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)

            writer.append_data(canvas)
        writer.close()

        video_dir = os.path.abspath(video_dir)
        print(f"Video saved to {video_path}")

        if group_id_all is not None:
            self.splats["sh0"] = sh0_origin



def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    # Use DeformRunner when single_finetune requires drag-based deformation
    if cfg.single_finetune and not cfg.finetune_with_only_rgb:
        raise Exception("Use DeformRunner.train_drag in deformsplat_trainer.py")
    else:
        runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for i, ckpt in enumerate(ckpts):
            if "clustered" not in ckpt:
                ckpts[i]["splats"] = cluster_largest(ckpts[i]["splats"])
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        # runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
        if cfg.single_finetune:

            if cfg.finetune_with_only_rgb:
                runner.cfg.max_steps = 1001
                runner.train()
                runner.eval(step=1001)
                       
                if cfg.render_traj_simple:
                    if cfg.data_name == "diva360":
                        render_traj_path = "diva360_interp"
                    elif cfg.data_name == "DFA":
                        render_traj_path = "dfa_interp"
                    runner.cfg.render_traj_path = render_traj_path
                    runner.render_traj(step=step, group_id_all=None, video_path=cfg.video_path)

            else:
                start_time = time.time()  # 시작 시간 기록
                runner.train_drag()
                end_time = time.time()    # 종료 시간 기록

                elapsed_time = end_time - start_time
                wandb.log({"drag_time": elapsed_time})
                
                if cfg.render_traj_simple:
                    if cfg.data_name == "diva360":
                        render_traj_path = "diva360_interp"
                    elif cfg.data_name == "DFA":
                        render_traj_path = "dfa_interp"
                    runner.cfg.render_traj_path = render_traj_path
                    runner.render_traj(step=step, group_id_all=None, video_path=cfg.video_path)
                    runner.make_motion_video(idx=0, threhold_early_stop=1e-5, scheduler_step=500, min_rigid_coef=0)
        

        if cfg.render_traj_all:
            # breakpoint()
            if cfg.data_name == "diva360":
                traj_path_list = ["diva360_interp"]  # "interp", "ellipse", "spiral"
            elif cfg.data_name == "DFA":
                traj_path_list = ["dfa_interp"]  # "interp", "ellipse", "spiral"
            for render_traj_path in traj_path_list:
                runner.cfg.render_traj_path = render_traj_path

                # video_dir = f"./output_video_traj/{cfg.data_name}"
                video_dir = cfg.result_dir
                os.makedirs(video_dir, exist_ok=True)
                video_paths = [
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_init.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_final.mp4"
                ]
                try:
                    group_ids = [
                        None,
                        ckpts[0]["group_id_all_init"],
                        ckpts[0]["group_id_all"],
                    ]
                    for i in range(3):
                        group_id_all = group_ids[i]
                        video_path = video_paths[i]
                        runner.render_traj(step=step, group_id_all=group_id_all, video_path=video_path)
                        runner.cfg.sh_degree = 0
                
                    # concat three video by width 
                    video_path = f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_concat.mp4"
                    os.system(f'ffmpeg -y -i "{video_paths[0]}" -i "{video_paths[1]}" -i "{video_paths[2]}" -filter_complex hstack=inputs=3 "{video_path}"')
                    print(f"video_path: {video_path}")

                    for video_path in video_paths:
                        os.remove(video_path)
                except:
                    # group_id_all = group_ids[i]
                    # video_path = video_paths[i]
                    video_path = f"{video_dir}/traj_{cfg.object_name}.mp4"
                    runner.render_traj(step=step, group_id_all=None, video_path=video_path)


    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)



def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    # Use DeformRunner when single_finetune requires drag-based deformation
    if cfg.single_finetune and not cfg.finetune_with_only_rgb:
        runner = DeformRunner(local_rank, world_rank, world_size, cfg)
    else:
        raise Exception("DeformSPlat is only supported")
        
    if cfg.ckpt is None:
        raise Exception("3DGS ckpt is required. Please train 3dgs with simple_trainer.py")
    
    ckpts = [
        torch.load(file, map_location=runner.device, weights_only=True)
        for file in cfg.ckpt
    ]
    for i, ckpt in enumerate(ckpts):
        if "clustered" not in ckpt:
            ckpts[i]["splats"] = cluster_largest(ckpts[i]["splats"])
    for k in runner.splats.keys():
        runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
    step = ckpts[0]["step"]

    runner.train_deformsplat()

    if cfg.render_traj_simple:
        if cfg.data_name == "diva360":
            render_traj_path = "diva360_interp"
        elif cfg.data_name == "DFA":
            render_traj_path = "dfa_interp"
        runner.cfg.render_traj_path = render_traj_path
        runner.render_traj(step=step, group_id_all=None, video_path=cfg.video_path)
        
        runner.make_motion_video(idx=0, threhold_early_stop=5e-6, scheduler_step=800, min_rigid_coef=0.01)
        
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    # cli(main, cfg, verbose=True)

    # Logger
    main(0, 0, 1, cfg)

