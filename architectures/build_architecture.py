"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""
######################################################################
def build_architecture(config):
    """
    Dynamically selects and builds the model architecture based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model_name and other parameters.

    Returns:
        model (torch.nn.Module): The initialized model architecture.

    Raises:
        ValueError: If the specified model_name is not supported.
    """
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("model_name not specified in the config.")

    # Define a mapping between model names and their corresponding import paths
    model_mapping = {
        "segformer3d": "segformer3d.build_segformer3d_model",
        "nnFormer": "nowmodel.build_seg_model",
        "3DUnet": "nowmodel.build_seg_model",
        "TransBTS": "nowmodel.build_seg_model",
        "UNETR": "nowmodel.build_seg_model",
        "SwinUNETR": "nowmodel.build_seg_model",
        "3DUX-net": "nowmodel.build_seg_model",
        "RepUXNET": "nowmodel.build_seg_model",
        "DeformUXNET": "nowmodel.build_seg_model",
        "segNow": "nowmodel.build_seg_model",
    }

    # Check if the model_name is supported
    if model_name not in model_mapping:
        raise ValueError(
            f"Model '{model_name}' not supported. Update build_architecture.py to add support."
        )

    # Dynamically import and build the model
    module_path, function_name = model_mapping[model_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    build_model_function = getattr(module, function_name)

    # Build and return the model
    return build_model_function(config)
