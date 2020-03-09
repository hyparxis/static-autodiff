#include "Eigen/Dense"

namespace adiff {

using NUM = double;

// TODO: pass in -1 to deduce input size from previous layer?
// TODO: assert that forward has been called before backward?

/*******************************************************************************
 LINEAR
*******************************************************************************/
// n = input dimension
// m = output dimension
template<int n, int m>
class LinearBase {
public:
  static constexpr int out_dim = m;

  using Matrix = Eigen::Matrix<NUM, m, n>;

  using VectorN = Eigen::Matrix<NUM, n, 1>; // in
  using VectorM = Eigen::Matrix<NUM, m, 1>; // out

  VectorM forward(const VectorN& x) {
    return W_ * x + b_;
  }

  VectorM operator() (const VectorN& x) {
    return forward(x);
  }

  // NOTE: most nn libraries use row vectors for
  // affine transforms (i.e. x * W + b) so make sure
  // W is not transposed if loading weights externally
  void setWeights(const Matrix& W,
                  const VectorM& b) {
    W_ = W;
    b_ = b;
  }

protected:
  Matrix W_;
  VectorM b_;
};

// Non-root specialization
template<int n, int m, typename Layer = void>
class Linear : public LinearBase<n, m> {
public:
  static constexpr int root_dim = Layer::root_dim;
  using Jacobian = Eigen::Matrix<NUM, m, root_dim>;

  Linear(Layer& parent) : parent_(parent) {}

  Jacobian backward() {
    return this->W_ * parent_.backward();
  }

private:
  Layer& parent_;
};

// Root specialization
template<int n, int m>
class Linear<n, m, void> : public LinearBase<n, m> {
public:
  static constexpr int root_dim = n;
  using Jacobian = Eigen::Matrix<NUM, m, root_dim>;

  Jacobian backward() {
    return this->W_;
  }
};


/*******************************************************************************
 TANH
*******************************************************************************/
// n = input/output dimension
template<int n>
class TanhBase {
public:
  static constexpr int out_dim = n;

  using Vector = Eigen::Matrix<NUM, n, 1>;

  Vector forward(const Vector& x) {
    x_ = x;
    return x_.array().tanh();
  }

  Vector operator() (const Vector& x) {
    return forward(x);
  }

protected:
  Vector x_;
};

// Non-root specialization
template<int n, typename Layer = void>
class Tanh : public TanhBase<n> {
public:
  static constexpr int root_dim = Layer::root_dim;
  using Jacobian = Eigen::Matrix<NUM, n, root_dim>;

  Tanh(Layer& parent) : parent_(parent) {};

  Jacobian backward() {
    // use auto to keep expression template from evaluating prematurely
    auto tmp = (1 - this->x_.array().tanh().pow(2)).matrix().asDiagonal();
    return tmp * parent_.backward();
  }

private:
  Layer& parent_;
};

// Root specialization
template<int n>
class Tanh<n, void> : public TanhBase<n> {
public:
  static constexpr int root_dim = n;
  using Jacobian = Eigen::Matrix<NUM, n, root_dim>;

  Jacobian backward() {
    return (1 - this->x_.array().tanh().pow(2)).matrix().asDiagonal();
  }
};
 
/*******************************************************************************
 Sum
*******************************************************************************/
template<int n>
class SumBase {
public:
  using Vector = Eigen::Matrix<NUM, n, 1>;

  NUM forward(const Vector& x) {
    return x.sum();
  }

  NUM operator() (const Vector& x) {
    return forward(x);
  }
};

// Non-root specialization
template<int n, typename Layer = void>
class Sum : public SumBase<n> {
public:
  static constexpr int root_dim = Layer::root_dim;
  using Jacobian = Eigen::Matrix<NUM, 1, root_dim>;

  Sum(Layer& parent) : parent_(parent) {}

  Jacobian backward() {
    return Eigen::Matrix<NUM, 1, n>::Ones() * parent_.backward();
  }

private:
  Layer& parent_;
};

// Root specialization
template<int n>
class Sum<n, void> : public SumBase<n> {
public:
  static constexpr int root_dim = n;
  using Jacobian = Eigen::Matrix<NUM, 1, root_dim>;

  Jacobian backward() {
    return Jacobian::Ones();
  }
};

/*******************************************************************************
 NN
*******************************************************************************/
// An example neural network
class NN {
public:
  static constexpr int in_dim = 4;
  static constexpr int out_dim = 2;
  static constexpr int h_dim = 64;

  using VectorOut = Eigen::Matrix<NUM, out_dim, 1>;
  using VectorIn = Eigen::Matrix<NUM, in_dim, 1>;
  using Jacobian = Eigen::Matrix<NUM, out_dim, in_dim>;

  NN() {
    // initialize weights here
  }

  VectorOut forward(const VectorIn& x) {
    return l3(l2(l1(l0(x))));
  }

  Jacobian backward() {
    return l3.backward();
  }

private:
  adiff::Linear<in_dim, h_dim>                l0;
  adiff::Tanh<h_dim, decltype(l0)>            l1{l0};
  adiff::Linear<h_dim, out_dim, decltype(l1)> l2{l1};
  adiff::Tanh<out_dim, decltype(l2)>          l3{l2};
};


} // namespace adiff
