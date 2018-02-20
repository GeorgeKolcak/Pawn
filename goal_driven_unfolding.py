import parametrised_unfolding


class SoftLimitParameterContext(parametrised_unfolding.ParameterContext):
    def __init__(self, graph = 0):
        super().__init__(graph)

        self.soft_limit = self.lattice.copy()

    def empty(self):
        return (not self.soft_limit) or self.soft_limit.empty()

    def bounds(self):
        return self.soft_limit

    def copy(self):
        copy = super().copy()

        copy.soft_limit = self.soft_limit.copy()

        return copy

    def limit(self, context, value):
        super().limit(context, value)

        self.soft_limit.limit(context, value)

    def limit_min(self, context, value):
        super().limit_min(context, value)

        self.soft_limit.limit_min(context, value)

    def limit_max(self, context, value):
        super().limit_max(context, value)

        self.soft_limit.limit_max(context, value)

    def soft_limit_min(self, context, value):
        self.soft_limit.limit_min(self, context.id, value)

    def soft_limit_max(self, context, value):
        self.soft_limit.limit_max(self, context.id, value)