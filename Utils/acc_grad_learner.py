from fastai.vision import *
from fastai.basic_train import BasicLearner


def convert(x):

    if not isinstance(x, nn.ModuleList):
        return x
    else:
        return nn.Sequential(*x)


def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None, checkpoint_segments=2)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]

    out = model(*xb)
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()


def fit(epochs:int, learn:BasicLearner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None,
        checkpoint_segments=2)->None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    assert len(learn.data.train_dl) != 0, f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler,
                                  checkpoint_segments=checkpoint_segments)
                if cb_handler.on_batch_end(loss): break

            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(learn.model, learn.data.valid_dl, loss_func=learn.loss_func,
                                    cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)


class AccumulateOptimWrapper(OptimWrapper):
    def step(self):           pass

    def zero_grad(self):      pass

    def real_step(self):      super().step()

    def real_zero_grad(self): super().zero_grad()


class OneCycleScheduler(LearnerCallback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."

    def __init__(self, learn: Learner, lr_max: float, moms: Floats = (0.95, 0.85), div_factor: float = 25.,
                 pct_start: float = 0.3, final_div: float = None, annealing_func=annealing_cos,
                 tot_epochs: int = None, start_epoch: int = None):
        super().__init__(learn)
        self.lr_max, self.div_factor, self.pct_start, self.final_div = lr_max, div_factor, pct_start, final_div
        if self.final_div is None: self.final_div = div_factor * 1e4
        self.moms = tuple(listify(moms, 2))
        if is_listy(self.lr_max): self.lr_max = np.array(self.lr_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs
        self.annealing_func = annealing_func

    def steps(self, *steps_cfg: StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Scheduler(step, n_iter, func=func)
                for (step, (n_iter, func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs: int, epoch: int, **kwargs: Any) -> None:
        "Initialize our optimization params based on our annealing schedule."
        res = {'epoch': self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs)
        n = len(self.learn.data.train_dl) * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n - a1
        self.phases = ((a1, self.annealing_func), (a2, self.annealing_func))
        low_lr = self.lr_max / self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, self.lr_max / self.final_div))
        self.mom_scheds = self.steps(self.moms, (self.moms[1], self.moms[0]))
        self.opt = self.learn.opt
        self.opt.lr, self.opt.mom = self.lr_scheds[0].start, self.mom_scheds[0].start
        self.idx_s = 0
        return res

    def jump_to_epoch(self, epoch: int) -> None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs: Any) -> None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:
            if self.idx_s >= len(self.lr_scheds): return {'stop_training': True, 'stop_epoch': True}
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        "Tell Learner to stop if the cycle is finished."
        if epoch > self.tot_epochs: return {'stop_training': True}


class AccGradLearner(Learner):

    def create_opt(self, lr: Floats, wd: Floats = 0.) -> None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = AccumulateOptimWrapper.create(self.opt_func, lr, self.layer_groups,
                                                 wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def fit(self, epochs:int, lr:Union[Floats,slice]=defaults.lr,
            wd:Floats=None, callbacks:Collection[Callback]=None,
            checkpoint_segments=2)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        if not getattr(self, 'opt', False): self.create_opt(lr, wd)
        else: self.opt.lr,self.opt.wd = lr,wd
        callbacks = [cb(self) for cb in self.callback_fns + listify(defaults.extra_callback_fns)] + listify(callbacks)
        if defaults.extra_callbacks is not None: callbacks += defaults.extra_callbacks
        fit(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks,
            checkpoint_segments=checkpoint_segments)

    def fit_one_cycle(self, cyc_len: int, max_lr: Union[Floats, slice] = defaults.lr,
                      moms: Tuple[float, float] = (0.95, 0.85), div_factor: float = 25., pct_start: float = 0.3,
                      final_div: float = None, annealing_func=annealing_cos,
                      wd: float = None, callbacks: Optional[CallbackList] = None, tot_epochs: int = None,
                      start_epoch: int = None,
                      checkpoint_segments=2) -> None:
        "Fit a model following the 1cycle policy."
        max_lr = self.lr_range(max_lr)
        callbacks = listify(callbacks)
        callbacks.append(OneCycleScheduler(self, max_lr, moms=moms, div_factor=div_factor, pct_start=pct_start,
                                           final_div=final_div, annealing_func=annealing_func,
                                           tot_epochs=tot_epochs, start_epoch=start_epoch))
        self.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks, checkpoint_segments=checkpoint_segments)

