import aesara.tensor as at


class StateSpaceLikelihoodGrad(at.Op):

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, model):
        self.out_func = model.f_loglike_grad

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        outputs[0][0] = self.out_func(theta)


class StateSpaceLikelihood(at.Op):

    itypes = [at.dvector]
    otypes = [at.dscalar]

    def __init__(self, model):
        self.out_func = model.f_loglike
        self.grad_loglike = StateSpaceLikelihoodGrad(model)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        outputs[0][0] = self.out_func(theta)

    def grad(self, inputs, output_grads):
        (theta,) = inputs
        return [output_grads[0] * self.grad_loglike(theta)]


class StateSpaceMean(at.Op):

    itypes = [at.dvector]
    otypes = [at.dtensor3]

    def __init__(self, model):
        self.out_func = model.f_y_hat

    def perform(self, node, inputs, outputs):
        (theta, ) = inputs
        outputs[0][0] = self.out_func(theta)


class StateSpaceCovariance(at.Op):

    itypes = [at.dvector]
    otypes = [at.dtensor3]

    def __init__(self, model):
        self.out_func = model.f_cov_hat

    def perform(self, node, inputs, outputs):
        (theta, ) = inputs
        outputs[0][0] = self.out_func(theta)
