# Create a simple feedforward neural network
using Flux
using Flux: @epochs
using Plots
using Distributions
using Random

# Create a model class to explore
P = 1
f(t, A, ϕ) = A .* sin.(P*t .+ ϕ)

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

function metatrain!(model, N::Int, α::Float64, β::Float64, K::Int)
  for i in range(1, N)
    println("Epoch: $i")
    # Sample some tasks
    tasks = sample(25);
    metaloss = [];
    θ = Flux.params(model)

    model_i = deepcopy(model);

    metaopt = Flux.Optimise.ADAM(θ, β)
    for (t, task) in enumerate(tasks)

      # Disconnect the model
      θ_i = deepcopy(θ)
      model_i = deepcopy(model)

      batchopt = Flux.Optimise.ADAM(θ_i, α)

      # Sample K data points
      x_s = rand(Uniform(-5, 5), K)
      data = [(map(x -> [x], x_s), task(x_s))];
      loss1(x, y) = mean(Flux.mse.(model_i.(x), y))
      l = loss1(data[1]...)
      Flux.Tracker.back!(loss1(data[1]...)); # Backpropagate through batchloss
      batchopt(); # Update

      # Resample
      x_s = rand(Uniform(-5, 5), K)
      data = [(map(x -> [x], x_s), task(x_s))];
      l = loss1(data[1]...) # Evaluate loss at these new params
      push!(metaloss, l);
    end

    metaopt = Flux.Optimise.runall(metaopt)
    l = sum(metaloss)
    Flux.Tracker.back!(l) # Backpropogate through metaloss
    metaopt() # Update
  end
end


model, θ = initialise()
loss1(x, y) = mean(Flux.mse.(model.(x), y))

metatrain!(model, 50000, 1e-3, 1e-3, 10)
#model = deepcopy(model_metatrained)
# Attempt K-shot learning
#model = deepcopy(model_metatrained)
tasks = sample(5)
task = tasks[1]

### Before update
x_test = -5:0.01:5
y_test = task.(x_test)
y_pred_0 = map(x -> x[1].data, model.(map(x-> [x], x_test)))


### Update

# Store model before updating
model_metatrained = deepcopy(model)

x_s = rand(Uniform(-5, 5), K)
data = [(map(x -> [x], x_s), task(x_s))]
metaopt = Flux.Optimise.ADAM(Flux.params(model), 1e-3)
@epochs 1 Flux.train!(loss1, data, metaopt)
x_test = -5:0.01:5
y_test = task.(x_test)
y_pred_1 = map(x -> x[1].data, model.(map(x-> [x], x_test)))


@epochs 9 Flux.train!(loss1, data, metaopt)
y_pred_10 = map(x -> x[1].data, model.(map(x-> [x], x_test)))

p = plot(x_test, y_test)
p = plot!(x_test, y_pred_0)
p = plot!(x_test, y_pred_1)
p = plot!(x_test, y_pred_10)
p = scatter!(x_s, task(x_s))


plot(p, label = ["True", "Before Update", "1 Update", "10 Updates", "Sampled Data"])
