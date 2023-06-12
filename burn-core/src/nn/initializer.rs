use burn_tensor::Shape;
use libm::sqrt;
use rand::distributions;

use crate::config::Config;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

use crate as burn;

/// Enum specifying with what values a tensor should be initialized
#[derive(Config, Debug, PartialEq)]
pub enum Initializer {
    /// Fills tensor with values drawn uniformly between specified values
    Uniform(f64, f64),
    /// Fills tensor with values drawn uniformly between -sqrt(1/fan_in) and sqrt(1/fan_in).
    NormalizedUniform,
    /// Fills tensor with values drawn from normal distribution with specified mean and std
    Normal(f64, f64),
    /// Fills tensor with specified value everywhere
    Constant(f64),
    /// Fills tensor with 1s everywhere
    Ones,
    /// Fills tensor with 0s everywhere
    Zeros,
    /// Fills tensor with values according to the uniform version of Xavier Glorot initialization described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierUniform(f64),
    /// Fills tensor with values according to the normal version of Xavier Glorot initialization described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierNormal(f64),
}

impl Initializer {
    pub fn init<B: Backend, const D: usize, S: Into<Shape<D>>>(&self, shape: S, with_bias: bool) -> (Tensor<B, D>,
                                                                                                     Option<Tensor<B, 1>>) {
        match self {
            Self::Uniform(a, b) => {
                let distribution = Distribution::Uniform((*a).elem::<B::FloatElem>(), (*b).elem::<B::FloatElem>());
                random_weight_bias(shape, distribution, with_bias)
            }
            Self::NormalizedUniform => {
                let distribution = normalized_uniform(shape);
                random_weight_bias(&shape, distribution, with_bias)
            }
            Self::Normal(mean, std) => {
                let distribution = Distribution::Normal(*mean, *std);
                random_weight_bias(shape, distribution, with_bias)
            }
            Self::Constant(value) => Tensor::<B, D>::zeros(shape) + *value, //TODO replace with fill()
            Self::Ones => Tensor::<B, D>::ones(shape),
            Self::Zeros => Tensor::<B, D>::zeros(shape),
            Self::XavierUniform(gain) => {
                let distribution = xavier_uniform(gain, shape);
                random_weight_bias(&shape, distribution, with_bias)
            }
            Self::XavierNormal(gain) => {
                let distribution = xavier_normal(gain, shape);
                random_weight_bias(&shape, distribution, with_bias)
            }
        }
    }
}

fn random_weight_bias<B: Backend, const D: usize, S: Into<Shape<D>>>(
    shape: S,
    distribution: Distribution<<B as Backend>::FloatElem>, with_bias: bool)
    -> (Tensor<B, D>, Option<Tensor<B, 1>>) {
    let weight = Tensor::<B, D>::random(shape, distribution);
    let bias = if with_bias {
        Some(Tensor::<B, 1>::random(shape.into().dims[0], distribution))
    } else {
        None
    };
    (weight, bias)
}

struct Fans {
    fan_in: usize,
    fan_out: usize,
}

impl Fans {
    pub fn new<const D: usize>(shape: &Shape<D>) -> Self {
        assert!(
            D >= 2,
            "Can't calculate fan_in and fan_out on shapes smaller than 2"
        );

        let receptive_field_size: usize = shape.dims.iter().skip(2).product();
        let d: Vec<usize> = shape.dims.iter().take(2).map(|fan| fan * receptive_field_size).collect();
        Fans { fan_in: d[1], fan_out: d[0] }
    }

    pub fn sum(&self) -> usize {
        self.fan_in + self.fan_out
    }
}

fn normalized_uniform<B: Backend, const D: usize>(shape: &Shape<D>) -> Distribution<<B as Backend>::FloatElem> {
    let fan_in = Fans::new(shape).fan_in;
    let k = 1. / sqrt(fan_in as f64);
    Distribution::Uniform((-k).elem::<B::FloatElem>(), k.elem::<B::FloatElem>())
}


fn xavier_uniform<B: Backend, const D: usize>(
    gain: &f64,
    shape: &Shape<D>,
) -> Distribution<<B as Backxend>::FloatElem> {
    let fan_sum = Fans::new(shape).sum();
    let a = gain * sqrt(6.0 / fan_sum as f64);
    Distribution::Uniform((-a).elem::<B::FloatElem>(), a.elem::<B::FloatElem>())
}

fn xavier_normal<B: Backend, const D: usize>(
    gain: &f64,
    shape: &Shape<D>,
) -> Distribution<<B as Backend>::FloatElem> {
    let fan_sum = Fans::new(shape).sum();
    let std = gain * sqrt(2.0 / fan_sum as f64);
    Distribution::Normal(0.0, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn_tensor::Data;

    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn initializer_uniform_init() {
        TB::seed(0);

        let (a, b) = (0.0, 1.0);
        let (uniform, _): (Tensor<TB, 4>, _) = Initializer::Uniform(a, b).init([2, 2, 2, 2], false);

        uniform.into_data().assert_within_range(a..b);
    }

    #[test]
    fn initializer_normal_init() {
        // seed random generator
        TB::seed(0);
        let (mean, std) = (0.0, 1.0);
        let (normal, _): (Tensor<TB, 1>, _) = Initializer::Normal(mean, std).init([1000], false);
        let (var_act, mean_act) = normal.var_mean(0);

        let var_act: f32 = var_act.into_scalar().elem();
        let mean_act: f32 = mean_act.into_scalar().elem();

        assert!(
            var_act > 0.9 && var_act < 1.1,
            "Expected variance to be between 1.0 += 0.1, but got {var_act}"
        );
        assert!(
            mean_act > -0.1 && mean_act < 0.1,
            "Expected mean to be between 0.0 += 0.1, but got {mean_act}"
        );
    }

    #[test]
    fn initializer_constant_init() {
        let value = 5.0;
        let (constants, _): (Tensor<TB, 4>, _) = Initializer::Constant(value).init([2, 2, 2, 2], false);
        constants
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([value as f32 * 16.0]), 3);
    }

    #[test]
    fn initializer_zeros_init() {
        let (zeros, _): (Tensor<TB, 4>, _) = Initializer::Zeros.init([2, 2, 2, 2], false);
        zeros
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([0.0]), 3);
    }

    #[test]
    fn initializer_ones_init() {
        let (ones, _): (Tensor<TB, 4>, _) = Initializer::Ones.init([2, 2, 2, 2], false);
        ones.sum()
            .to_data()
            .assert_approx_eq(&Data::from([16.0]), 3);
    }

    #[test]
    fn initializer_xavier_uniform_init() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let bound = gain * sqrt(6. / (fan_in + fan_out) as f64);
        let (xavier_uniform, _): (Tensor<TB, 2>, _) =
            Initializer::XavierUniform(gain).init([fan_in, fan_out], false);

        xavier_uniform
            .into_data()
            .assert_within_range(-bound..bound);
    }

    #[test]
    fn initializer_xavier_uniform_init_with_receptive_field() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let (rec_field_1, rec_field_2) = (3, 4);

        let bound = gain * sqrt(6. / ((fan_in + fan_out) * rec_field_1 * rec_field_2) as f64);
        let (xavier_uniform, _): (Tensor<TB, 4>, _) =
            Initializer::XavierUniform(gain).init([fan_in, fan_out, rec_field_1, rec_field_2], false);

        xavier_uniform
            .into_data()
            .assert_within_range(-bound..bound);
    }

    #[test]
    fn initializer_xavier_normal_init() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;

        let expected_var = (gain * sqrt(2. / (fan_in as f64 + fan_out as f64))).powf(2.);
        let (xavier_normal, _): (Tensor<TB, 2>, _) = Initializer::XavierNormal(gain).init([fan_in, fan_out], false);
        let (actual_vars, actual_means) = xavier_normal.var_mean(0);

        for i in 0..fan_out {
            let actual_var = actual_vars.to_data().value[i] as f64;
            let actual_mean = actual_means.to_data().value[i] as f64;

            assert!(
                (expected_var - actual_var).abs() <= 0.1,
                "Expected variance to be between {expected_var} += 0.1, but got {actual_var}"
            );
            assert!(
                (expected_mean - actual_mean).abs() <= 0.1,
                "Expected mean to be between {expected_mean} += 0.1, but got {actual_mean}"
            );
        }
    }
}
