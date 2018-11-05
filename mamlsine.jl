# Create a simple feedforward neural network
using Flux
using Flux: @epochs
using Plots
using Distributions
using Random

# Create a model class to explore
P = 1
f(t, A, ϕ) = A .* sin.(P*2*π*t .+ ϕ)

function initialise()
  m = Chain(
   Dense(1, 40, Flux.relu),
   Dense(40, 40, Flux.relu),
   Dense(40, 1, identity),
  )
  θ = Flux.params(m)
  return m, θ
end

function sample(N)
  # Sample sets of A and θ:
  p_train = zip(rand(Uniform(0.1, 5), N), rand(Uniform(0, π), N))
  map(p -> t -> f(t, p[1], p[2]), p_train)
end

function metatrain!(model, N, opt1, opt2, loss1)
  for i in range(1, N)
    println("Epoch: $i")
    # Sample some tasks
    tasks = sample(10)
    θ_i = deepcopy(Flux.params(model))
    _model = deepcopy(model)
    metaloss = []
    for (t, task) in enumerate(tasks)
      θ = deepcopy(θ_i) # Reinitialize the parameters
      model = deepcopy(_model)
      # Sample K data points
      x_s = rand(Uniform(-5, 5), K)
      data = [(map(x -> [x], x_s), task(x_s))]
      l = loss1(data[1]...)
      #Flux.train!(loss1, data, opt1)
      Flux.Tracker.back!(loss1(data[1]...)) # Backpropogate through metaloss
      opt1() # Update


      # Resample
      x_s = rand(Uniform(-5, 5), K)
      data = [(map(x -> [x], x_s), task(x_s))]

      # Evaluate loss at these new parameters
      l = loss1(data[1]...)
      push!(metaloss, l)
    end
    model = deepcopy(_model)
    opt = Flux.Optimise.runall(opt2)
    l = sum(metaloss)
    Flux.Tracker.back!(l) # Backpropogate through metaloss
    opt() # Update
  end
end


K = 10
α = 1e-2
β = 1e-2
model, θ = initialise()
loss1(x, y) = mean(Flux.mse.(model.(x), y))
opt_1 = Flux.Optimise.ADAM(θ, α)
opt_2 = Flux.Optimise.ADAM(θ, β)

metatrain!(model, 1000, opt_1, opt_2, loss1)

# Attempt K-shot learning
tasks = sample(5)
task = tasks[1]

### Before update
x_test = -5:0.01:5
y_test = task.(x_test)
y_pred = map(x -> x[1].data, model.(map(x-> [x], x_test)))
scatter(x_test, y_test)
scatter!(x_test, y_pred)

### Update

x_s = rand(Uniform(-5, 5), K)
data = [(map(x -> [x], x_s), task(x_s))]
Flux.train!(loss1, data, opt1)
x_test = -5:0.01:5
y_test = task.(x_test)
y_pred = map(x -> x[1].data, model.(map(x-> [x], x_test)))

scatter(x_test, y_test)
scatter!(x_test, y_pred)
