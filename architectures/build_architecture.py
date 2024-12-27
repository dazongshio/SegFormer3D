def build_seg_model(config):
    """
    Generic function to build segmentation models based on configuration.

    Args:
        config (dict): Configuration dictionary containing 'name' and specific parameters.

    Returns:
        model (torch.nn.Module): The initialized model.

    Raises:
        ValueError: If the specified model_name is invalid or not supported.
    """
    model_name = config['model']['name']
    model_parameters = config['model'][model_name]

    # Ensure the model name is valid
    if model_name not in [
        'segformer3d', 'nnFormer', 'TransBTS', 'UNETR', 'SwinUNETR',
        'UXNet3D', 'RepUXNET', 'DeformUXNET', 'segNow', '3DUnet', 'nn-UNet'
    ]:
        raise ValueError(f"Invalid model name: {model_name}. Supported models are defined in the configuration.")

    # Define per-model construction logic
    if model_name == 'segformer3d':
        from segformer3d import SegFormer3D
        model = SegFormer3D(
            in_channels=model_parameters["in_channels"],
            sr_ratios=model_parameters["sr_ratios"],
            embed_dims=model_parameters["embed_dims"],
            patch_kernel_size=model_parameters["patch_kernel_size"],
            patch_stride=model_parameters["patch_stride"],
            patch_padding=model_parameters["patch_padding"],
            mlp_ratios=model_parameters["mlp_ratios"],
            num_heads=model_parameters["num_heads"],
            depths=model_parameters["depths"],
            decoder_head_embedding_dim=model_parameters["decoder_head_embedding_dim"],
            num_classes=model_parameters["num_classes"],
            decoder_dropout=model_parameters["decoder_dropout"],
        )

    elif model_name == 'nnFormer':
        from nnFormer import nnFormer
        model = nnFormer(
            input_channels=model_parameters["in_channels"],
            crop_size=model_parameters["patch_dim"],
            embedding_dim=model_parameters["embedding"],
            patch_size=model_parameters["patch_kernel_siz4"],
            window_size=model_parameters["window_size"],
            num_heads=model_parameters["num_heads"],
            depths=model_parameters["depths"],
            num_classes=model_parameters["num_classes"],
            conv_op=model_parameters["conv_op"],
            deep_supervision=model_parameters["deep_supervision"],
        )

    elif model_name == 'UXNet3D':
        from architectures.UXNet3D import UXNet3D

        model = UXNet3D(
            in_chans=model_parameters["in_channels"],
            out_chans=model_parameters["num_classes"],
            depths=model_parameters["depths"],
            feat_size=model_parameters["feat_size"],
            drop_path_rate=model_parameters["drop_path_rate"],
            layer_scale_init_value=model_parameters["layer_scale_init_value"],
            hidden_size=model_parameters["hidden_size"],
            norm_name=model_parameters["norm_name"],
            conv_block=model_parameters["conv_block"],
            res_block=model_parameters["res_block"],
            spatial_dims=model_parameters["spatial_dims"],
        )

    elif model_name == 'TransBTS':
        from TransBTS import BTS
        model = BTS(
            img_dim = model_parameters["img_dim"],
            patch_dim=model_parameters["patch_dim"],
            num_channels=model_parameters["in_channels"],
            num_classes=model_parameters["num_classes"],
            embedding_dim=model_parameters["embedding"],
            num_heads=model_parameters["num_heads"],
            num_layers=model_parameters["num_layers"],
            hidden_dim=model_parameters["hidden_dim"],
            dropout_rate=model_parameters["dropout_rate"],
            attn_dropout_rate=model_parameters["attn_dropout_rate"],
            conv_patch_representation=model_parameters["conv_patch_representation"],
            positional_encoding_type=model_parameters["positional_encoding_type"],
        )

    elif model_name == 'UNETR':
        from monai.networks.nets.unetr import UNETR
        model = UNETR(
            img_size=model_parameters["img_size"],
            in_channels=model_parameters["in_channels"],
            out_channels=model_parameters["num_classes"],
            num_heads=model_parameters["num_heads"],
            feature_size=model_parameters["feature_size"],
            norm_name=model_parameters["norm_name"],
            hidden_size=model_parameters["hidden_size"],
            mlp_dim=model_parameters["mlp_dim"],
            dropout_rate=model_parameters["dropout_rate"],
            pos_embed=model_parameters["pos_embed"],
            conv_block=model_parameters["conv_block"],
            res_block=model_parameters["res_block"],
            spatial_dims=model_parameters["spatial_dims"],
            qkv_bias=model_parameters["qkv_bias"],
            save_attn=model_parameters["save_attn"],
        )

    elif model_name == 'SwinUNETR':
        from monai.networks.nets.swin_unetr import SwinUNETR
        model = SwinUNETR(
        img_size=model_parameters["img_size"],
        in_channels=model_parameters["in_channels"],
        out_channels=model_parameters["num_classes"],
        depths=model_parameters["depths"],
        num_heads=model_parameters["num_heads"],
        feature_size=model_parameters["feature_size"],
        norm_name=model_parameters["norm_name"],
        drop_rate=model_parameters["drop_rate"],
        attn_drop_rate=model_parameters["attn_drop_rate"],
        dropout_path_rate=model_parameters["dropout_path_rate"],
        normalize=model_parameters["normalize"],
        use_checkpoint=model_parameters["use_checkpoint"],
        spatial_dims=model_parameters["spatial_dims"],
        downsample = model_parameters["downsample"],
        use_v2 = model_parameters["use_v2"],

        )

    elif model_name == 'RepUXNET':
        from RepUXNET import REPUXNET
        model = REPUXNET(
            in_chans=model_parameters["in_channels"],
            out_chans=model_parameters["num_classes"],
            depths=model_parameters["depths"],
            feat_size=model_parameters["feat_size"],
            drop_path_rate=model_parameters["drop_path_rate"],
            layer_scale_init_value=model_parameters["layer_scale_init_value"],
            hidden_size=model_parameters["hidden_size"],
            norm_name=model_parameters["norm_name"],
            conv_block=model_parameters["conv_block"],
            res_block=model_parameters["res_block"],
            spatial_dims=model_parameters["spatial_dims"],
            ks=model_parameters["ks"],
            a=model_parameters["a"],
            deploy=model_parameters["deploy"],

        )

    elif model_name == 'DeformUXNET':
        from deformUXNET import DEFORMUXNET
        model = DEFORMUXNET(
            in_chans=model_parameters["in_channels"],
            out_chans=model_parameters["num_classes"],
            depths=model_parameters["depths"],
            feat_size=model_parameters["feat_size"],
            drop_path_rate=model_parameters["drop_path_rate"],
            layer_scale_init_value=model_parameters["layer_scale_init_value"],
            hidden_size=model_parameters["hidden_size"],
            norm_name=model_parameters["norm_name"],
            conv_block=model_parameters["conv_block"],
            res_block=model_parameters["res_block"],
            spatial_dims=model_parameters["spatial_dims"],
        )

    # elif model_name == 'segNow':
    #     from nowmodel import now
    #     model = SegNow(
    #         in_channels=config[model_name]["in_channels"],
    #         num_classes=config[model_name]["num_classes"],
    #         backbone=config[model_name]["backbone"],
    #     )

    # elif model_name == '3DUnet' or model_name == 'nn-UNet':
    #     model = UNet3D(
    #         in_channels=config[model_name]["in_channels"],
    #         out_channels=config[model_name]["out_channels"],
    #         final_sigmoid=config[model_name]["final_sigmoid"],
    #     )

    else:
        raise ValueError(f"Model '{model_name}' is not yet implemented.")

    return model
