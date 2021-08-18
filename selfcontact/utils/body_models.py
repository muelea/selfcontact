import smplx

def get_bodymodels(
        model_path, 
        model_type, 
        device, 
        batch_size=1, 
        num_pca_comps=12
    ):

    models = {}

    # model parameters for self-contact optimization
    model_params = dict(
        batch_size=batch_size,
        model_type=model_type,
        create_body_pose=True,
        create_transl=False,
        create_betas=False,
        create_global_orient=False,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=True,
        num_pca_comps=num_pca_comps,
        return_full_pose=True,
    )
    
    # create smplx model per gender
    for gender in ['male', 'female', 'neutral']:
        models[gender] = smplx.create(
            model_path=model_path,
            gender=gender,
            **model_params
        ).to(device)
    
    return models