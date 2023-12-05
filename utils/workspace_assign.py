from config import cfg


def name_assign():
    base_name = "mode_{}".format(cfg.MODEL.mode)

    if cfg.MODEL.mode == "edge":
        extra_name = assign_edge()
    else:
        extra_name = assign_ende()

    extra_name += "_batch_{}".format(cfg.SOLVER.image_per_batch)
    if cfg.SOLVER.max_iters == 0:
        extra_name += "_epochs_{}".format(cfg.SOLVER.num_epochs)
    else:
        extra_name += "_iters_{}".format(cfg.SOLVER.max_iters)

    train_name = base_name + extra_name

    return train_name


def assign_edge():
    base_name = "_er_{:.1f}".format(cfg.SOLVER.edge_r)
    return base_name


def assign_ende():
    base_name = "code_{}".format(cfg.MODEL.latent_len)

    extra_name = "_{}".format(cfg.MODEL.name)
    if cfg.MODEL.select_latent:
        extra_name += "_slct"
    # if cfg.MODEL.select_layer:
    #     extra_name += "_slctlayer"
    if cfg.MODEL.domainada:
        extra_name += "_dclf"
    if cfg.MODEL.aeweight:
        extra_name += "_aeweight"
    if cfg.MODEL.pretrained:
        extra_name += "_aepre"
    extra_name += "_latent_{}".format(cfg.MODEL.latent_len)
    extra_name += "_lbdfd_{:.3f}".format(cfg.SOLVER.lambda_fd)

    return base_name + extra_name


if __name__ == "__main__":
    pass
