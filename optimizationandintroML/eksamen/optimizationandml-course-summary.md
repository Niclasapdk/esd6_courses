# Optimization and Machine Learning - Complete Course Summary

## Part I: Optimization Fundamentals

### 1.1 The Optimization Problem

The fundamental optimization problem seeks to find the minimum value of a function over a specified domain:

```
min_{x∈R} f(x)
```

**Key Components:**
- **Objective/Cost Function**: `f: dom(f) ⊆ ℝⁿ → ℝ` - The function we want to minimize
  - In machine learning, often called the "loss function"
  - Maps from n-dimensional input space to real values
  
- **Feasible Region**: `R ⊆ dom(f)` - The set of allowed solutions
  - Defines the constraints of the problem
  - Can be the entire domain (unconstrained) or a subset (constrained)

- **Domain**: `dom(f)` - The largest open set where f is defined

**Problem Notation:**
- Standard form: `min_{x∈R} f(x)`
- Alternative notations:
  - `min{f(x) | x ∈ R}`
  - `min f(x) s.t. x ∈ R` (s.t. = "subject to")
  - Unconstrained: `min f(x)` when R = dom(f)

**Related Problems:**
- Finding minimizers: `argmin_{x∈R} f(x)` - returns the x values that minimize f
- Maximization: `max_{x∈R} f(x) = -min_{x∈R} -f(x)`

### 1.2 Types of Minimizers and Minima

Understanding different types of optimal points is crucial for optimization:

#### Local Minimizers
- **Weak local minimizer**: `f(x) ≥ f(x')` for all x in neighborhood of x'
- **Strong local minimizer**: `f(x) > f(x')` for all x ≠ x' in neighborhood of x'
- Only considers points near x', not the entire feasible region

#### Global Minimizers
- **Weak global minimizer**: `f(x) ≥ f(x')` for all x ∈ R
- **Strong global minimizer**: `f(x) > f(x')` for all x ∈ R, x ≠ x'
- The true optimal solution over entire feasible region

**Important:** In general optimization, a function may have:
- Multiple local minima
- Multiple global minima (if weak)
- At most one strong global minimum

### 1.3 Essential Calculus for Optimization

#### Continuity and Differentiability

**C¹ Functions** (Continuously Differentiable):
- All first-order partial derivatives exist and are continuous
- Required: `∂f/∂xᵢ` exists and is continuous for all i = 1,...,n
- Ensures smoothness for gradient-based methods

**C² Functions** (Twice Continuously Differentiable):
- All first and second-order partial derivatives exist and are continuous
- Required: `∂²f/∂xᵢ∂xⱼ` exists and is continuous for all i,j
- Necessary for second-order methods (Newton's method)

#### Gradient, Hessian, and Jacobian

**Gradient** ∇f(x) - First-order information:
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ ∈ ℝⁿ
```
- Points in direction of steepest ascent
- Perpendicular to level curves/surfaces
- Zero at stationary points

**Hessian** H_f(x) - Second-order information:
```
H_f(x) = [∂²f/∂xⱼ∂xᵢ]ᵢⱼ ∈ ℝⁿˣⁿ
```
- Symmetric matrix for C² functions
- Captures curvature information
- Eigenvalues determine local behavior

**Jacobian** J_Φ(x) - For vector-valued functions:
```
J_Φ(x) = [∂Φᵢ/∂xⱼ]ᵢⱼ ∈ ℝᵐˣⁿ for Φ: ℝⁿ → ℝᵐ
```
- Generalizes gradient to vector outputs
- Rows are gradients of component functions

#### Taylor Approximations

Linear approximation (using gradient):
```
f(x + d) ≈ f(x) + ∇f(x)ᵀd
```

Quadratic approximation (using Hessian):
```
f(x + d) ≈ f(x) + ∇f(x)ᵀd + ½dᵀH_f(x)d
```

### 1.4 Optimality Conditions

#### First-Order Necessary Conditions

For x' to be a local minimizer:

**Interior Points** (x' ∈ int(R)):
```
∇f(x') = 0
```
- Points where gradient vanishes are called **stationary** or **critical** points
- Not all stationary points are minimizers!

**Boundary Points**:
```
∇f(x')ᵀd ≥ 0 for all feasible directions d ∈ F(x')
```
- Gradient points "outward" from feasible region

#### Second-Order Necessary Conditions

For x' to be a local minimizer with x' ∈ int(R):
```
∇f(x') = 0 and H_f(x') ⪰ 0 (positive semi-definite)
```
- All eigenvalues of Hessian must be non-negative
- Rules out saddle points and maxima

#### Second-Order Sufficient Conditions

If x' ∈ int(R) and:
```
∇f(x') = 0 and H_f(x') ≻ 0 (positive definite)
```
Then x' is a **strong local minimizer**
- All eigenvalues of Hessian are positive
- Guarantees local optimality

## Part II: Convex Analysis

### 2.1 Convex Sets

A set is convex if it contains all line segments between its points:

**Definition**: R_c ⊂ ℝⁿ is convex if:
```
αx₁ + (1-α)x₂ ∈ R_c for all x₁, x₂ ∈ R_c and α ∈ [0,1]
```

**Intuition**: No "dents" or "holes" - rubber band around set touches everywhere

#### Important Convex Sets

**Vector Spaces and Subspaces**:
- ℝⁿ (entire space)
- Range(A) = {z | z = Ax} (column space)
- Null(A) = {x | Ax = 0} (null space)

**Affine Sets**:
- Lines: {z | z = θx₁ + (1-θ)x₂, θ ∈ ℝ}
- Hyperplanes: {x | aᵀx = b}
- Affine subspaces: {x | Ax = b}

**Half-spaces and Polyhedra**:
- Half-space: {x | aᵀx ≤ b}
- Polyhedron: {x | Ax ≤ b} (intersection of half-spaces)
- Polytope: Bounded polyhedron

**Convex Cones**:
- Set C where αx + βy ∈ C for all x,y ∈ C and α,β > 0
- Example: Positive orthant ℝⁿ₊

#### Operations Preserving Convexity

1. **Intersection**: If R₁, R₂ convex, then R₁ ∩ R₂ convex
2. **Affine transformation**: If R convex and f(x) = Ax + b, then f(R) convex
3. **Convex combination**: Weighted average with non-negative weights summing to 1

### 2.2 Convex Functions

A function is convex if its graph lies below any chord:

**Definition**: f is convex on convex set R_c if:
```
f(αx₁ + (1-α)x₂) ≤ αf(x₁) + (1-α)f(x₂)
```
for all x₁, x₂ ∈ R_c and α ∈ [0,1]

**Strictly Convex**: Strict inequality for x₁ ≠ x₂

#### Characterizations of Convex Functions

**Epigraph Characterization**:
- f convex ⟺ epi f = {(x,μ) | μ ≥ f(x)} is convex set

**First-Order Characterization** (C¹ functions):
```
f convex ⟺ f(x) ≥ f(x') + ∇f(x')ᵀ(x-x') for all x,x'
```
- Function lies above all tangent hyperplanes

**Second-Order Characterization** (C² functions):
```
f convex ⟺ H_f(x) ⪰ 0 for all x
```
- Non-negative curvature everywhere

#### Operations Preserving Convexity

1. **Non-negative combination**: αf₁ + βf₂ convex if α,β ≥ 0
2. **Pointwise maximum**: max{f₁(x), f₂(x)} convex
3. **Supremum**: sup_{y∈Y} f(x,y) convex in x
4. **Composition**: h(g(x)) convex under certain monotonicity conditions

### 2.3 Common Convex Functions in Machine Learning

#### Basic Functions
- **Linear**: f(x) = wᵀx + b
- **Quadratic**: f(x) = ½xᵀQx + bᵀx + c (if Q ⪰ 0)
- **Exponential**: f(x) = exp(x)
- **Negative logarithm**: f(x) = -log(x) for x > 0

#### Norms (All Convex)
- **L1 norm**: ‖x‖₁ = Σ|xᵢ| (promotes sparsity)
- **L2 norm squared**: ‖x‖₂² = xᵀx (smooth, differentiable)
- **L∞ norm**: ‖x‖∞ = max|xᵢ|
- **General Lp norm**: ‖x‖_p = (Σ|xᵢ|^p)^(1/p)

#### Loss Functions
- **Mean Squared Error**: (1/n)Σ(yᵢ - xᵢ)²
- **Mean Absolute Error**: (1/n)Σ|yᵢ - xᵢ|
- **Hinge Loss**: max(0, 1 - y·(wᵀx + b))
- **Cross-Entropy**: -Σyᵢlog(pᵢ(x))

#### Regularization
- **Lasso (L1)**: λ‖w‖₁
- **Ridge (L2)**: λ‖w‖₂²
- **Elastic Net**: α‖w‖₁ + (1-α)‖w‖₂²

#### Activation Functions
- **ReLU**: max(0, x) (convex but not differentiable at 0)
- **Log-sum-exp**: log(Σexp(xᵢ)) (smooth approximation of max)

**Non-convex examples**: Sigmoid, tanh, swish

## Part III: Convex Optimization

### 3.1 Properties of Convex Optimization Problems

When both objective function and feasible region are convex:

**Key Properties**:
1. **Every local minimum is global**: No getting stuck in suboptimal valleys
2. **Set of minimizers is convex**: If multiple optimal solutions, they form convex set
3. **First-order conditions sufficient**: ∇f(x')ᵀ(x-x') ≥ 0 for all x ∈ R ⟹ x' optimal
4. **Efficient algorithms exist**: Polynomial-time solutions for many problems

**Maximizers of Convex Functions**:
- Always lie on boundary of feasible region
- Interior points cannot be maximizers

### 3.2 Least Squares Problem

Fundamental problem in machine learning and statistics:

**Problem Formulation**:
```
min_x ‖b - Ax‖₂²
```
- A ∈ ℝᵐˣⁿ: Design/feature matrix
- b ∈ ℝᵐ: Observation/target vector
- x ∈ ℝⁿ: Parameter vector to find

**Solution via Normal Equations**:
```
AᵀAx = Aᵀb
```
- Always has solution (may not be unique)
- If A has full column rank: x̂ = (AᵀA)⁻¹Aᵀb = A†b
- A† is the Moore-Penrose pseudoinverse

**Geometric Interpretation**:
- Ax̂ is orthogonal projection of b onto Range(A)
- Residual r = b - Ax̂ is orthogonal to Range(A)
- Minimizes squared distance to column space

**Connection to Linear Regression**:
- Each row of A represents features for one data point
- b contains target values
- x contains learned weights/parameters

## Part IV: Gradient Methods

### 4.1 Steepest Descent (Gradient Descent)

Most fundamental iterative optimization algorithm:

**Basic Algorithm**:
```
x_{k+1} = x_k + α_k d_k
where d_k = -∇f(x_k)
```

**Key Ideas**:
- Move in direction of negative gradient (steepest descent)
- Step size α_k crucial for convergence
- Simple but can be slow for ill-conditioned problems

#### Step Size Selection Methods

**1. Exact Line Search**:
```
α_k ∈ argmin_{α≥0} f(x_k + αd_k)
```
- Optimal step in given direction
- Often impractical to compute exactly

**2. Golden Section Search**:
- Maintains interval [a,b] containing minimum
- Uses golden ratio φ ≈ 1.618 for efficiency
- Reduces interval by constant factor each iteration

**3. Backtracking Line Search**:
```
Start with α = 1
While f(x + αd) > f(x) + ĉαα∇f(x)ᵀd:
    α = βα
```
- Armijo condition ensures sufficient decrease
- Parameters: ĉα ∈ (0, 0.5), β ∈ (0, 1)

**4. Fixed Step Size**:
- α_k = α̂ (constant learning rate)
- Simple but requires tuning
- May diverge if too large

#### Convergence Analysis

**Convergence Rate**:
```
f(x_{k+1}) - f(x*) ≤ ((1-r)/(1+r))² [f(x_k) - f(x*)]
```
where r = λ_min/λ_max (condition number)

**Implications**:
- Fast convergence when eigenvalues similar (r ≈ 1)
- Slow convergence for ill-conditioned problems (r ≈ 0)
- Zigzagging behavior near minimum

### 4.2 Newton-Raphson Method

Second-order method using curvature information:

**Algorithm**:
```
d_k = -H_f(x_k)⁻¹∇f(x_k)
x_{k+1} = x_k + α_k d_k
```

**Key Features**:
- Uses Hessian to adjust for curvature
- Quadratic convergence near minimum
- More expensive per iteration (O(n³) for Hessian inverse)

**Modifications for Non-Convex Problems**:
- If H_f not positive definite, modify:
  - Add multiple of identity: H' = H + εI
  - Use H' = (1/(1+β))H + (β/(1+β))I
- Ensures descent direction

**Comparison with Gradient Descent**:
- Fewer iterations needed
- Each iteration more expensive
- Better for ill-conditioned problems
- Requires second derivatives

### 4.3 Gauss-Newton Method

Specialized for nonlinear least squares:

**Problem**: Solve F(x) = 0 by minimizing ‖F(x)‖₂²

**Key Approximation**:
```
∇f(x) = 2J_F(x)ᵀF(x)
H_f(x) ≈ 2J_F(x)ᵀJ_F(x)
```
- Ignores second-order terms in F
- Only requires Jacobian of F, not full Hessian

**Algorithm**:
```
d_k = -(J_F(x_k)ᵀJ_F(x_k))⁻¹J_F(x_k)ᵀF(x_k)
```

**Applications**:
- Nonlinear regression
- Neural network training (special cases)
- Computer vision (bundle adjustment)

### 4.4 Stochastic Gradient Descent (SGD)

Essential for large-scale machine learning:

**Problem Setup**:
```
min_θ (1/N)Σᵢ₌₁ᴺ ℓ(h_θ(xᵢ), yᵢ)
```
- N can be millions/billions
- Full gradient expensive

**SGD Algorithm**:
1. Randomly shuffle data
2. Partition into mini-batches
3. Update using gradient on mini-batch:
   ```
   θ_{k+1} = θ_k - α∇f_{batch}(θ_k)
   ```

**Key Concepts**:
- **Mini-batch size**: Trade-off between variance and computation
- **Learning rate schedule**: Often decrease over time
- **Momentum**: Add fraction of previous update
- **Shuffling**: Prevents cycling, reduces bias

**Advantages**:
- Computational efficiency
- Can escape shallow local minima
- Online learning capability

**Challenges**:
- High variance in updates
- Hyperparameter tuning
- Convergence criteria less clear

## Part V & VI: Constrained Optimization

### 5.1 Problem Formulation

General constrained optimization problem:

```
min f(x)
s.t. a(x) = 0    (equality constraints)
     c(x) ≥ 0    (inequality constraints)
```

Where:
- a: ℝⁿ → ℝᵖ (p equality constraints)
- c: ℝⁿ → ℝᵍ (q inequality constraints)
- Notation: c(x) ≥_e 0 means componentwise inequality

**Feasible Region**:
```
R = {x ∈ ℝⁿ | a(x) = 0, c(x) ≥ 0}
```

### 5.2 Regular Points and Constraint Qualifications

**Regular Point Definition**:
Point x' is regular if gradients of active constraints are linearly independent:
- For equality: ∇a₁(x'), ..., ∇a_p(x') linearly independent
- For inequality: Include ∇c_j(x') where c_j(x') = 0 (active)

**Active vs Inactive Constraints**:
- Active: c_i(x') = 0 (constraint is "tight")
- Inactive: c_i(x') > 0 (constraint not binding)
- Only active constraints affect local optimality

**Linear Constraints Special Case**:
- Ax = b regular if rank(A) = p
- Consistency: rank([A b]) = rank(A)
- No redundancy needed for regularity

### 5.3 Lagrange Multipliers and Lagrangian

**Lagrangian Function**:
```
L(x, λ, μ) = f(x) - λᵀa(x) - μᵀc(x)
```
- λ ∈ ℝᵖ: Multipliers for equality constraints
- μ ∈ ℝᵍ: Multipliers for inequality constraints
- μ ≥ 0 required

**Geometric Interpretation**:
- At optimum, ∇f lies in span of constraint gradients
- Can't improve objective without violating constraints
- Multipliers are "prices" for relaxing constraints

### 5.4 Karush-Kuhn-Tucker (KKT) Conditions

**First-Order Necessary Conditions**:
If x* is local minimum and regular point, then ∃ λ*, μ* such that:

1. **Stationarity**:
   ```
   ∇f(x*) = Σλᵢ*∇aᵢ(x*) + Σμⱼ*∇c_j(x*)
   ```

2. **Primal Feasibility**:
   ```
   a(x*) = 0, c(x*) ≥ 0
   ```

3. **Dual Feasibility**:
   ```
   μ* ≥ 0
   ```

4. **Complementary Slackness**:
   ```
   μᵢ*cᵢ(x*) = 0 for all i
   ```
   (Either μᵢ = 0 or cᵢ(x*) = 0)

**For Convex Problems**: KKT conditions are also sufficient!

### 5.5 Second-Order Conditions

**Bordered Hessian of Lagrangian**:
```
∇²ₓL(x, λ, μ) = H_f(x) - ΣλᵢH_{aᵢ}(x) - ΣμᵢH_{cᵢ}(x)
```

**Second-Order Necessary**:
```
dᵀ∇²ₓL(x*, λ*, μ*)d ≥ 0
```
for all d in null space of active constraint gradients

**Second-Order Sufficient**:
```
dᵀ∇²ₓL(x*, λ*, μ*)d > 0
```
for all d ≠ 0 in critical cone

### 5.6 Constraint Transformations

**Inequality to Equality** (using slack variables):
```
c(x) ≥ 0  ⟺  c(x) - s² = 0, s ∈ ℝᵍ
```

**Equality to Inequality**:
```
a(x) = 0  ⟺  a(x) ≥ 0 and -a(x) ≥ 0
```

**Box Constraints**:
```
l ≤ x ≤ u  ⟺  x - l ≥ 0 and u - x ≥ 0
```

## Key Exam Preparation Points

### Conceptual Understanding
1. **Convexity Recognition**: Identify convex sets/functions quickly
2. **Optimality Conditions**: Know when to apply which conditions
3. **Algorithm Selection**: Choose appropriate method for problem type
4. **Constraint Handling**: Transform between constraint types

### Computational Skills
1. **Gradient/Hessian Calculation**: For common functions
2. **KKT System Setup**: Write conditions for given problem
3. **Line Search Implementation**: Understand termination criteria
4. **Convergence Analysis**: Estimate rates for different methods

### Common Pitfalls to Avoid
1. Confusing necessary vs sufficient conditions
2. Forgetting regularity assumptions
3. Missing complementary slackness in KKT
4. Incorrect Hessian for constrained problems

### Problem-Solving Strategy
1. Check convexity first
2. Identify constraint structure
3. Verify regularity
4. Apply appropriate optimality conditions
5. Select suitable algorithm

This comprehensive summary covers all major topics in optimization for machine learning. Each section builds on previous concepts, creating a coherent framework for understanding modern optimization methods.

# Machine Learning - Complete Course Summary

## Part VII: Introduction to ML & Bayesian Decision Theory

### 7.1 What is Machine Learning?

**Definition**: Machine Learning is the field concerned with using optimization techniques to develop strategies that leverage data to optimize performance criteria.

**Key Applications**:
- Object recognition from images
- Next word prediction in sentences
- Dimensionality reduction for data visualization
- Large Language Models (LLMs)

**Related Fields**:
- **Statistics**: Statistical estimation addresses similar problems; most ML strategies are statistical
- **Optimization**: ML uses optimization principles with data to optimize performance criteria

### 7.2 Types of Machine Learning

#### Supervised Learning
Given input data with output labels, predict output for test data:
- **Regression**: Continuous outputs (e.g., CO2 concentration prediction)
- **Classification**: Discrete outputs (e.g., cancer image classification)

#### Unsupervised Learning
Given only input data (no labels), find structure:
- **Dimensionality Reduction**: Project high-D data to low-D (e.g., 4D→3D for visualization)
- **Density Estimation**: Estimate probability distributions
- **Clustering**: Discover patterns (e.g., sleep pattern types)

### 7.3 The Bias-Variance Tradeoff

**Underfitting (High Bias)**:
- Model too simple to capture data patterns
- Poor performance on both training and test data
- Example: Linear model for non-linear data

**Overfitting (High Variance)**:
- Model too complex, memorizes training data
- Great training performance, poor test performance
- Example: High-degree polynomial for simple data

**Key Insight**: Very flexible models have low bias but high variance; rigid models have high bias but low variance.

#### Cross-Validation
Technique to evaluate model performance reliably:
- **k-Fold CV**: Split data into k folds
  - Train on k-1 folds, validate on 1 fold
  - Repeat k times, average results
  - Reduces dependence on single train/validation split

**Data Splitting Best Practice**:
```
Total Data = Training Data + Validation Data + Test Data
- Training: Fit model parameters
- Validation: Tune hyperparameters, check overfitting
- Test: Final evaluation (use only once!)
```

### 7.4 Uncertainty in Machine Learning

**Sources of Uncertainty**:
- Measurement errors (noisy sensors)
- Limited data
- Model assumptions
- Inherent randomness

**Why It Matters**: "55% chance of disease A" is more informative than "You have disease A"

### 7.5 Probability Review for ML

#### Key Concepts

**Random Variables**: Functions mapping outcomes to numbers
- Discrete: Probability Mass Function (PMF)
- Continuous: Probability Density Function (PDF)

**Fundamental Rules**:
```
Conditional: P(X|Y) = P(X,Y)/P(Y)
Product: P(X,Y) = P(X|Y)P(Y)
Marginal: P(X) = Σ_y P(X,Y) or ∫ P(X,Y)dy
Bayes: P(X|Y) = P(Y|X)P(X)/P(Y)
```

**Expectation**:
```
E[f(X)] = Σ_x p_X(x)f(x) (discrete)
E[f(X)] = ∫ p_X(x)f(x)dx (continuous)
```

### 7.6 Bayesian Decision Theory

Framework for optimal decisions under uncertainty:

**Components**:
- **Outcomes**: Y ∈ Y (e.g., disease states)
- **Observations**: X ∈ X (e.g., test results)
- **Actions**: a ∈ A (e.g., treat/don't treat)
- **Loss Function**: L(y,a) - cost of action a when truth is y

**Decision Process**:
1. Observe data X = x
2. Compute posterior: p(Y|X) = p(X|Y)p(Y)/p(X)
3. Calculate expected loss: R(a|x) = E[L(Y,a)|X=x]
4. Choose optimal action: a* = argmin_a R(a|x)

#### 0-1 Loss (Classification)
```
L(y,a) = {0 if a=y (correct), 1 if a≠y (wrong)}
```
Optimal decision: **Maximum A Posteriori (MAP)**
```
a*(x) = argmax_y p(Y=y|X=x) = argmax_y p(X=x|Y=y)p(Y=y)
```

#### Reject Option
Add action "reject" with cost λ ∈ [0,1]:
```
a*(x) = {
  argmax_y p(Y=y|X=x)  if max_y p(Y=y|X=x) > 1-λ
  reject                otherwise
}
```

**Medical Example**:
- High cost for missing disease → lower classification threshold
- Reject option → additional tests when uncertain

## Part VIII: Parametric and Nonparametric Methods

### 8.1 Parametric Methods

**Definition**: Models with fixed, finite set of parameters θ independent of data size.

#### Maximum Likelihood Estimation (MLE)
Find parameters maximizing data likelihood:
```
θ_MLE = argmax_θ ∏_{n=1}^N p(x_n|θ)
```

**Gaussian MLE Example**:
```
μ_MLE = (1/N)Σ x_n (sample mean)
σ²_MLE = (1/N)Σ(x_n - μ_MLE)² (biased variance)
σ²_unbiased = (1/(N-1))Σ(x_n - μ_MLE)² (unbiased)
```

#### Maximum A Posteriori (MAP)
Include prior knowledge p(θ):
```
θ_MAP = argmax_θ p(θ|X) = argmax_θ p(X|θ)p(θ)
```

**Gaussian MAP with Gaussian Prior**:
```
μ_MAP = (Nσ²_prior)/(Nσ²_prior + σ²) × μ_MLE + σ²/(Nσ²_prior + σ²) × μ_prior
```
Balances data evidence with prior belief.

### 8.2 Naive Bayes Classifier

**Problem**: Full covariance matrix has O(d²) parameters for d features.

**Naive Assumption**: Features conditionally independent given class:
```
p(x|c) = ∏_{i=1}^d p(x_i|c)
```

**Benefits**:
- Reduces parameters from O(d²) to O(d) per class
- Works well even when independence assumption violated
- Efficient for high-dimensional data (e.g., text)

**Parameter Estimation**:
```
For each class c and feature i:
μ_c,i = (1/N_c)Σ_{n:y_n=c} x_n,i
σ²_c,i = (1/(N_c-1))Σ_{n:y_n=c} (x_n,i - μ_c,i)²
p(c) = N_c/N
```

### 8.3 Nonparametric Methods

**Definition**: Model complexity grows with data size; often store all training data.

#### Histograms
Fixed bin width h:
```
p̂(x) = (# points in bin around x)/(N × h)
```
Or with kernel K:
```
p̂(x) = (1/Nh)Σ_n K((x-x_n)/h)
```

#### k-Nearest Neighbors (k-NN)
Fix number of neighbors k, vary region size:
```
p̂(x) = k/(N × V_k(x))
```
where V_k(x) is volume containing k nearest neighbors.

**k-NN Classification**:
```
p̂(c|x) = k_c/k (fraction of k neighbors in class c)
Classify as: argmax_c k_c
```

**Properties**:
- No training phase (lazy learning)
- Flexible decision boundaries
- Sensitive to distance metric choice
- Computationally expensive at test time

### 8.4 Model Evaluation Metrics

**Classification Metrics**:
```
Accuracy = (TP + TN)/(TP + TN + FP + FN)
Sensitivity = TP/(TP + FN) (True Positive Rate)
Specificity = TN/(TN + FP) (True Negative Rate)
```

**Confusion Matrix**: Summarizes all prediction outcomes
```
           Predicted
Actual    Pos    Neg
  Pos     TP     FN
  Neg     FP     TN
```

## Part X: Linear Regression

### 10.1 Linear Regression Model

**General Form**:
```
g(x) = Σ_{m=0}^{M-1} w_m φ_m(x) = w^T φ(x)
```
- w: weight/coefficient vector
- φ(x): basis functions (chosen to make data linear in feature space)

**Common Basis Functions**:
- Polynomial: φ_m(x) = x^m
- Fourier: φ_m(x) = sin(2πfx), cos(2πfx)
- Gaussian: φ_m(x) = exp(-||x-μ_m||²/2σ²)
- Identity: φ_m(x) = x_m (standard linear regression)

### 10.2 Least Squares = Maximum Likelihood

**Model with Gaussian Noise**:
```
y = w^T φ(x) + ε, where ε ~ N(0,σ²)
```

**MLE Solution**:
```
w_MLE = (Φ^T Φ)^{-1} Φ^T y
```
where Φ is the design matrix with rows φ(x_n)^T.

This is identical to minimizing squared error:
```
min_w (1/2)||y - Φw||²
```

**Geometric Interpretation**: 
- Projects y onto column space of Φ
- Φw_MLE is closest point in Range(Φ) to y

### 10.3 Regularization

**Motivation**: Prevent overfitting when:
- Many basis functions (high M)
- Limited data (small N)

#### Ridge Regression (L2)
```
J(w) = (1/2)||y - Φw||² + (λ/2)||w||²
```

**Closed-form Solution**:
```
w_ridge = (Φ^T Φ + λI)^{-1} Φ^T y
```

**MAP Interpretation**: Gaussian prior on weights
```
p(w) ~ N(0, τ²I), with λ = σ²/τ²
```

#### Lasso (L1)
```
J(w) = (1/2)||y - Φw||² + λ||w||₁
```

**Properties**:
- Induces sparsity (exactly zero coefficients)
- Automatic feature selection
- No closed-form solution

**MAP Interpretation**: Laplace prior on weights

#### Comparison
| Property | Ridge (L2) | Lasso (L1) |
|----------|------------|------------|
| Solution | Closed-form | Iterative |
| Sparsity | No | Yes |
| Feature Selection | No | Yes |
| Prior | Gaussian | Laplace |

**Elastic Net**: Combines both: α||w||₁ + (1-α)||w||²₂

### 10.4 Practical Considerations

**Normalization**: Essential for numerical stability
```
Centered: x̃ = x - mean(x)
Scaled: x̃ = (x - mean(x))/std(x)
```

**Choosing λ**: Cross-validation to balance bias-variance

## Part XI: Linear Classification

### 11.1 From Regression to Classification

**Basic Linear Classifier**:
```
ŷ = sign(w^T φ(x))
```
- Decision boundary: {x : w^T φ(x) = 0}
- w is perpendicular to decision boundary

**Multi-class Extension**:
```
ŷ = argmax_c w_c^T φ(x)
```
Creates piecewise linear decision boundaries.

### 11.2 Learning Algorithms

#### Least Squares Classification
Encode labels as one-hot vectors:
```
If y_n = c, then ỹ_n = [0,...,1,...,0]^T (1 at position c)
```
Apply standard least squares:
```
W_LS = (Φ^T Φ)^{-1} Φ^T Ỹ
```

**Limitations**:
- No probabilistic interpretation
- Sensitive to outliers
- Can give poor decision boundaries

#### Perceptron Algorithm
**Model**: ŷ = sign(w^T x) for y ∈ {-1, +1}

**Perceptron Criterion**:
```
E_p(w) = Σ_{n∈M} -w^T x_n y_n
```
where M = misclassified points.

**Update Rule** (SGD with learning rate η):
```
w_{t+1} = w_t + η x_n y_n (for misclassified x_n)
```

**Properties**:
- Guaranteed convergence if linearly separable
- No convergence if not separable
- No probabilistic output
- No simple multi-class extension

### 11.3 Probabilistic Classification: Logistic Regression

**Binary Classification Model**:
```
p(y=1|x,w) = σ(w^T φ(x)) = 1/(1 + exp(-w^T φ(x)))
p(y=0|x,w) = 1 - σ(w^T φ(x))
```

**Key Properties**:
- Outputs probabilities ∈ [0,1]
- Smooth, differentiable
- Decision boundary at p = 0.5 (i.e., w^T φ(x) = 0)

**Maximum Likelihood**:
```
L(w) = Π_n σ(w^T φ(x_n))^{y_n} (1-σ(w^T φ(x_n)))^{1-y_n}
```

**Log-likelihood** (negative = cross-entropy loss):
```
ℓ(w) = Σ_n [y_n log σ(w^T φ(x_n)) + (1-y_n)log(1-σ(w^T φ(x_n)))]
```

**Gradient** (for gradient ascent):
```
∇ℓ(w) = Σ_n [y_n - σ(w^T φ(x_n))]φ(x_n) = Φ^T(y - σ(Φw))
```

### 11.4 Multi-class Classification: Softmax

**Softmax Function**:
```
p(y=c|x,w) = exp(w_c^T φ(x))/Σ_k exp(w_k^T φ(x))
```

**Properties**:
- Generalizes sigmoid to K classes
- Outputs valid probability distribution
- Decision: argmax_c p(y=c|x,w)

**Cross-Entropy Loss**:
```
L = -Σ_n Σ_c ỹ_{n,c} log p(y=c|x_n,w)
```
where ỹ_{n,c} is one-hot encoding.

### 11.5 Connection to Naive Bayes

For Gaussian Naive Bayes with equal variances:
```
log[p(y=1|x)/p(y=0|x)] = w^T x + w_0
```
where w = (μ₁ - μ₀)/σ².

**Insight**: Logistic regression learns the discriminative function directly without assuming Gaussian distributions.

## Part XII: Introduction to Neural Networks

### 12.1 Motivation and Architecture

**Limitation of Linear Models**: Require hand-engineered features φ(x).

**Solution**: Learn feature representations automatically!

#### Single Neuron (Perceptron)
```
a = Σ_m w_m x_m (pre-activation)
z = h(a) (activation)
```

Common activation functions:
- Identity: h(a) = a (for regression)
- Sign: h(a) = sign(a) (for classification)
- Sigmoid: h(a) = 1/(1+e^{-a})
- ReLU: h(a) = max(0,a)
- Tanh: h(a) = (e^a - e^{-a})/(e^a + e^{-a})

#### Multi-Layer Perceptron (MLP)
Stack neurons in layers:
- Input layer: raw features
- Hidden layers: learned representations
- Output layer: predictions

### 12.2 Expressive Power

**Universal Approximation Theorem**: 
Any continuous function on [0,1]^d can be approximated arbitrarily well by an MLP with one hidden layer (given enough neurons).

**Why Deep Networks?**:
1. Some functions require exponentially fewer neurons with depth
2. Hierarchical feature learning
3. Parameter sharing across outputs

**Example**: Parity function requires exponential neurons in shallow network but linear in deep network.

### 12.3 Training Neural Networks

**Loss Function**:
```
L(w) = (1/N)Σ_n L_n(w)
```
Common losses:
- MSE: L_n = (1/2)||y_n - o_n||²
- Cross-entropy: L_n = -Σ_i y_{n,i} log o_{n,i}

**Optimization**: Gradient descent variants
```
w ← w - η∇L(w)
```

### 12.4 Backpropagation Algorithm

Efficient computation of gradients using chain rule.

**Key Steps**:

1. **Forward Pass**: Compute activations layer by layer
   ```
   a_j^(ℓ) = Σ_i w_{ji}^(ℓ) z_i^(ℓ-1)
   z_j^(ℓ) = h(a_j^(ℓ))
   ```

2. **Output Layer Sensitivity**:
   ```
   δ_j^(L) = h'(a_j^(L)) × ∂L/∂z_j^(L)
   ```

3. **Hidden Layer Sensitivities** (backward):
   ```
   δ_j^(ℓ) = h'(a_j^(ℓ)) × Σ_k w_{kj}^(ℓ+1) δ_k^(ℓ+1)
   ```

4. **Weight Gradients**:
   ```
   ∂L/∂w_{ji}^(ℓ) = δ_j^(ℓ) z_i^(ℓ-1)
   ```

**Special Cases**:
- Regression + MSE + identity activation: δ_j^(L) = o_j - y_j
- Classification + cross-entropy + softmax: δ_j^(L) = o_j - y_j

### 12.5 Practical Considerations

#### Vanishing/Exploding Gradients
**Problem**: Gradients become very small or large in deep networks.

**Causes**:
- Sigmoid/tanh saturate (gradient ≈ 0)
- Poor weight initialization

**Solutions**:
- ReLU activation (no saturation for positive inputs)
- Proper initialization (Xavier, He)
- Batch/Layer normalization
- Gradient clipping
- Skip connections (ResNet)

#### Non-differentiable Activations
- **Sign function**: Use smooth approximation (tanh) or set derivative = 1
- **ReLU at 0**: Convention is h'(0) = 0

#### Other Optimizers
Beyond vanilla SGD:
- **Momentum**: Accumulate velocity
- **Adam**: Adaptive learning rates per parameter
- **RMSprop**: Normalize by gradient magnitude

## Part XIII: PCA & SVD

### 13.1 Motivation for Dimensionality Reduction

**Why Reduce Dimensions?**
- Data visualization (high-D → 2D/3D)
- Computational efficiency
- Noise reduction
- Feature extraction
- Avoid curse of dimensionality

**Key Questions**:
1. Which directions preserve most information?
2. How much information is lost?

### 13.2 Principal Component Analysis (PCA)

**Goal**: Find orthogonal directions that maximize variance (minimize reconstruction error).

#### Mathematical Formulation

**Setup**:
1. Center data: x_n^(c) = x_n - x̄
2. Project onto directions u₁, ..., u_K
3. Minimize reconstruction error

**Optimization Problem**:
```
max_u u^T C u subject to ||u|| = 1
```
where C = (1/N)X^(c)(X^(c))^T is the covariance matrix.

**Solution**: u₁, ..., u_K are top K eigenvectors of C.

#### PCA Algorithm
1. Center data: X^(c) = X - x̄
2. Compute covariance: C = (1/N)X^(c)(X^(c))^T
3. Eigendecomposition: C = QΛQ^T
4. Select top K eigenvectors: U = [q₁ ... q_K]
5. Project: Z = U^T X^(c)
6. Reconstruct: X̂ = UZ + x̄

**Properties of Principal Components**:
- Orthogonal: u_i^T u_j = 0 for i ≠ j
- Ordered by variance: λ₁ ≥ λ₂ ≥ ... ≥ λ_d
- Uncorrelated projections: Cov[Z] = diag(λ₁, ..., λ_K)

### 13.3 Choosing Number of Components

**Individual Explained Variance** for PC_i:
```
λ_i / Σ_j λ_j
```

**Cumulative Explained Variance** for K components:
```
(Σ_{i=1}^K λ_i) / (Σ_{j=1}^d λ_j)
```

Common thresholds: 90%, 95%, or 99% variance retained.

### 13.4 Singular Value Decomposition (SVD)

**Definition**: For any matrix X ∈ ℝ^(d×N):
```
X = UΣV^T
```
where:
- U ∈ ℝ^(d×d): Left singular vectors (orthonormal)
- Σ ∈ ℝ^(d×N): Diagonal with singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0
- V ∈ ℝ^(N×N): Right singular vectors (orthonormal)

**Rank-1 Decomposition**:
```
X = Σ_{i=1}^r σ_i u_i v_i^T
```

### 13.5 Connection Between PCA and SVD

For centered data X^(c):
```
C = (1/N)X^(c)(X^(c))^T = (1/N)UΣ²U^T
```

Therefore:
- Eigenvectors of C = Left singular vectors of X^(c)
- Eigenvalues: λ_i = σ_i²/N
- PCA projections: Z = U^T X^(c) = ΣV^T

### 13.6 Advantages of SVD for PCA

1. **Numerical Stability**: Better handling of rank-deficient matrices
2. **Efficiency**: No need to form C explicitly (important for d >> N)
3. **Memory**: C requires O(d²) storage; SVD works with X directly

**Truncated SVD**: Keep only top K components
```
X ≈ U_K Σ_K V_K^T
```

### 13.7 Applications and Examples

**Image Compression**:
- Each image row/column as data point
- Truncated SVD gives compressed representation
- Storage: d×N → (d×K + K×K + K×N) for K << min(d,N)

**Face Recognition (Eigenfaces)**:
- Each face image as high-D vector
- Principal components capture face variations
- Project new faces onto "eigenface" basis

**Practical Insights**:
- First PCs often capture global patterns
- Later PCs capture finer details/noise
- Outliers can dominate first PCs (consider robust PCA)

## Key Exam Preparation Topics

### Core Concepts to Master

1. **Probability and Bayesian Inference**
   - Bayes rule and its applications
   - MAP vs MLE estimation
   - Posterior, likelihood, prior relationships

2. **Model Types and Trade-offs**
   - Parametric vs nonparametric
   - Discriminative vs generative
   - Linear vs nonlinear

3. **Optimization in ML**
   - Gradient descent variants
   - Convex objectives (logistic regression)
   - Non-convex (neural networks)

4. **Evaluation and Validation**
   - Cross-validation strategies
   - Confusion matrices and metrics
   - Bias-variance tradeoff

5. **Algorithm Specifics**
   - When to use each method
   - Computational complexity
   - Assumptions and limitations

### Common Pitfalls to Avoid

1. **Forgetting to normalize/center data** (crucial for PCA, neural networks)
2. **Using test data for anything except final evaluation**
3. **Ignoring probabilistic interpretations** (they often provide insight)
4. **Confusing generative and discriminative models**
5. **Not considering computational complexity** for large-scale problems

### Problem-Solving Strategy

1. **Identify problem type**: Classification/regression/unsupervised
2. **Check data characteristics**: Size, dimensionality, linearity
3. **Choose appropriate model**: Consider assumptions and complexity
4. **Validate properly**: Use held-out data, cross-validation
5. **Interpret results**: What do parameters mean? Is performance reasonable?

This summary integrates all major ML topics from your course, showing how optimization principles underlie all machine learning methods!