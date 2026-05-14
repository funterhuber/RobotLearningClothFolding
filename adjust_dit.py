    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # 1. Consolidate per-camera image keys into observation.images
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # 2. Remove action from batch (action is output, not input)
        if ACTION in batch:
            batch.pop(ACTION)

        # 3. Populate observation queues
        self._queues = populate_queues(self._queues, batch)

        # Stack n latest observations from the queue
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)

        return actions