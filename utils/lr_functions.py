import numpy as np
from collections import OrderedDict
from .logger import printlog


class LRFcts:
    def __init__(self, config: dict, lr_total_steps: int):
        self.base_lr = config['learning_rate']
        # fixme : an issue is caused when using stage-wise learning rate as base_lr is not the same
        # fixme : for every parameter group of the optimizer
        # if min_lr = 0 (which it is almost always) then this is not an issue
        self.base_val = 1.0
        self.lr_total_steps = lr_total_steps
        self.lr_fct = config['lr_fct']
        self.batchwise = config['lr_batchwise']
        self.lr_params = dict()
        if config['lr_params'] is not None:
            self.lr_params = config['lr_params']
        if self.lr_fct == 'piecewise_static':
            #  example entry in config['train']["piecewise_static_schedule"]: [[40,1],[50,0.1]]
            # if s<=40 ==> lr = learning_rate * 1 elif s<=50 ==> lr = learning_rate * 0.1
            assert 'piecewise_static_schedule' in self.lr_params
            assert isinstance(self.lr_params['piecewise_static_schedule'], list)
            assert self.lr_params['piecewise_static_schedule'][-1][0] == config['epochs'], \
                "piecewise_static_schedule's last phase must have first element equal to number of epochs " \
                "instead got: {} and {} respectively".format(config['piecewise_static_schedule'][-1][0], config['epochs'])

            piecewise_static_schedule = self.lr_params['piecewise_static_schedule']
            self.piecewise_static_schedule = OrderedDict() # this is essential, it has to be an ordered dict
            phase_prev = 0
            for phase in piecewise_static_schedule: # get ordered dict from list
                assert phase_prev < phase[0], ' piecewise_static_schedule must have increasing first elements per phase' \
                                              ' instead got phase_prev {} and phase {}'.format(phase_prev, phase[0])
                self.piecewise_static_schedule[phase[0]] = phase[1]

    def __call__(self, step: int):
        if step > self.lr_total_steps:
            printlog(f'warning learning rate scheduler at step {step} '
                     f'exceeds expected lr_total_steps {self.lr_total_steps}')
        if self.lr_fct == 'exponential':
            return self.lr_exponential(step)

        elif self.lr_fct == 'polynomial':
            return self.lr_polynomial(step, self.lr_total_steps)

        elif self.lr_fct == 'linear-warmup-polynomial':
            assert 'warmup_iters' in self.lr_params
            if step < self.lr_params['warmup_iters']-1:
                return self.linear_warmup(step)
            else:
                return self.lr_polynomial(step, self.lr_total_steps)

        elif self.lr_fct == 'linear-warmup-cosine':
            assert 'warmup_iters' in self.lr_params
            if step < self.lr_params['warmup_iters'] - 1:
                return self.linear_warmup(step)
            else:
                return self.lr_cosine(step, self.lr_total_steps)
        else:
            ValueError("Learning rate schedule without restarts'{}' not recognised.".format(self.lr_fct))

    def piecewise_static(self, step):
        # important this only works if self.piecewise_static_schedule is an ordered dict!
        for phase_end in self.piecewise_static_schedule.keys():
            lr = self.piecewise_static_schedule[phase_end]
            if step <= phase_end:
                return lr

    def linear_warmup(self, step: int):
        # step + 1 to account for step = 0 ... warmup_iters -1
        if self.lr_params.get('legacy', False):
            warmup_rate = self.lr_params.get('warmup_rate', 1e-6)
            lr = 1 - (1 - (step+1) / self.lr_params['warmup_iters']) * (1 - warmup_rate)
        else:
            T = self.lr_params['warmup_iters']
            start_lr = self.lr_params.get('warmup_start_mult', 0)  # lr@step=0
            end_lr = self.lr_params.get('warmup_end_mult', self.base_val)  # lr@step=warmup_iters-1
            # lr = rate * step + c
            # {"power": 1.0, "warmup_iters": 5000, "warmup_start": 1e-6, "warmup_end": 0.0001}
            rate = (end_lr - start_lr) / (T - 1)
            c = start_lr
            lr = rate * step + c
        # if step == 4999:
        #     printlog(f"peak lr reacherd {lr}")
        #     a = 1
        # warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        return lr

    #     def adjust_learning_rate(optimizer, epoch, args):
    #     """Decay the learning rate with half-cycle cosine after warmup"""
    #     if epoch < args.warmup_epochs:
    #         lr = args.lr * epoch / args.warmup_epochs
    #     else:
    #         lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
    #             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    #     for param_group in optimizer.param_groups:
    #         if "lr_scale" in param_group:
    #             param_group["lr"] = lr * param_group["lr_scale"]
    #         else:
    #             param_group["lr"] = lr
    #     return lr

    def lr_exponential(self, steps_current: int):
        base_val = self.base_val
        gamma = .98 if self.lr_params is None else self.lr_params
        lr = base_val * gamma ** steps_current
        return lr

    # def lr_polynomial(self, base_val: float, steps_current: int, max_steps: int):
    #     # max_steps - 1 to account for step = 0 ... max_steps -1
    #     # power = .9 if 'power' in self.lr_params else self.lr_params['power']
    #     power = self.lr_params.get('power', .9)
    #     # min_lr = self.lr_params['min_lr'] if 'min_lr' in self.lr_params else 0.0
    #     min_lr = self.lr_params.get('min_lr', 0.0)
    #     coeff = (1 - steps_current / (max_steps-1)) ** power
    #     lr = (base_val- min_lr) * coeff + min_lr
    #     return lr

    def lr_polynomial(self, steps_current: int, max_steps: int):
        # max_steps - 1 to account for step = 0 ... max_steps -1
        # power = .9 if 'power' in self.lr_params else self.lr_params['power']
        # important note: base_val is a multiplier of lr and not the actual lr (starting from base_lr)
        # so accordingly min_base_val is the ratio min_lr / base_lr
        if 'warmup' in self.lr_fct:
            elapsed_steps = self.lr_params.get("warmup_iters", 0.0)
        else:
            elapsed_steps = 0

        power = self.lr_params.get('power', 1.0)
        min_lr = self.lr_params.get('min_lr', 0.0)
        assert min_lr >= 0
        min_base_val = min_lr / self.base_lr   # fixme: see issue at start of file

        steps_ratio = (steps_current - elapsed_steps) / ((max_steps-1) - elapsed_steps)

        coeff = (1 - steps_ratio) ** power
        lr = (self.base_val - min_base_val) * coeff + min_base_val

        return max(lr, min_base_val)

    def lr_cosine(self, steps_current, max_steps):
        min_lr = self.lr_params.get('min_lr', 0.0)
        min_base_val = min_lr / self.base_lr
        if 'warmup' in self.lr_fct:
            elapsed_steps = self.lr_params.get("warmup_iters", 0.0)
        else:
            elapsed_steps = 0

        # args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            #             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        steps_ratio = (steps_current - elapsed_steps) / ((max_steps - 1) - elapsed_steps)
        lr = (self.base_val - min_base_val) * 0.5 * (1. + np.cos(np.pi * steps_ratio)) + min_base_val
        return lr


if __name__ == '__main__':
    def lr_exponential(base_val: float, steps_since_restart: int, steps_in_restart=None, gamma: int = .98):
        lr = base_val * gamma ** steps_since_restart
        return lr

    def lr_cosine(base_val, steps_since_restart, steps_in_restart):
        lr = base_val * 0.5 * (1. + np.cos(np.pi * steps_since_restart / steps_in_restart))
        return lr


    def linear_warmup(step: int):
        base_lr = 0.0001
        rate = 1e-6
        # step + 1 to account for step = 0 ... warmup_iters -1
        lr = 1 - (1 - (step+1) / 1500) * (1 - rate)
        # warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        return lr * base_lr

    def lr_polynomial( base_val: float, steps_current: int, max_steps: int):
        # max_steps - 1 to account for step = 0 ... max_steps -1
        power = 1.0
        min_lr = 0.0
        coeff = (1 - steps_current / (max_steps-1)) ** power
        lr = (base_val- min_lr) * coeff + min_lr
        return lr


    def linear_warmup_then_poly(step:int, total_steps):
        if step <= 1500 - 1:
            return linear_warmup(step)
        else:
            return lr_polynomial(0.0001, step, total_steps)




    # lr_start = 0.0001
    # T = 100
    # lrs = [lr_cosine(lr_start, step, T) for step in range(T)]
    # lrs_exp = [lr_exponential(lr_start, step % (T//4), T//4) for step in range(T)]
    #
    #
    #
    import matplotlib.pyplot as plt
    # plt.plot(lrs)
    # plt.plot(lrs_exp)
    T = 160401
    lrs_exp = [linear_warmup_then_poly(step, T) for step in range(T)]
    plt.plot(lrs_exp)
    plt.show()
    a = 1