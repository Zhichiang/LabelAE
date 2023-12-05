from config import cfg


def name_assign():
    base_name = "code_{}".format(cfg.MODEL.latent_len)

    extra_name = "_{}".format(cfg.MODEL.name)
    if cfg.MODEL.select_latent:
        extra_name += "_slct"
    if cfg.MODEL.select_layer:
        extra_name += "_slctlayer"
    if cfg.SOLVER.max_iters == 0:
        extra_name += "_epochs_{}".format(cfg.SOLVER.num_epochs)
    else:
        extra_name += "_iters_{}".format(cfg.SOLVER.max_iters)

    train_name = base_name + extra_name

    return train_name


if __name__ == "__main__":
    pass
