import attrs
import torch
import tqdm

@attrs.define(kw_only=True)
class Trainer:
    max_epochs: int = attrs.field(validator=attrs.validators.instance_of(int))
    val_every_n_epochs: int = attrs.field(validator=attrs.validators.instance_of(int))
    device: int = attrs.field(validator=attrs.validators.instance_of(int))
    callbacks: List[Callable] = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    loggers: List = attrs.field(factory=list, validator=attrs.validators.instance_of(list))

    @callbacks.validator
    def check_callbacks_are_callable(self, attribute, value):
        for callback in value:
            assert callable(callback), f"'{callback}' is not a callable function"

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        **kwargs: Any,
    ) -> None:
        model.to(device)
        optimisers = model.configure_optimizers()
        for epoch in enumerate(range(max_epochs)):
            if epoch % val_every_n_epochs == 0 and val_dataloader is not None:
                model.eval()
                for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                    loss = model.validation_step(batch, batch_idx)
            model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                for optimiser in optimisers:
                    optimizer.zero_grad()
                results = model.training_step(batch, batch_idx, **kwargs)
                results["loss"].backward()
                for optimiser in optimisers:
                    optimizer.step()

    def predict(
        self,
        model,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        pass
