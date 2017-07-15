New learners, policies, callbacks, environments, evaluation metrics or stopping
criteria need to implement the following functions.

# Learners
Learners that require only a (state, action, reward) triple and possibly the
next state and action should implement the first definition. If the learner is
also to be used with a NstepLearner one also needs to implement the second 
definition.
```@docs
update!
```

```@docs
act(learner, policy, state)
```

# Policies
```@docs
act(policy, values)
```

```@docs
getactionprobabilities
```

# Callbacks
```@docs
callback!
```

# [Environments](@id api_environments)
```@docs
interact!
```

```@docs
getstate
```

```@docs
reset!
```

# Evaluation Metrics
```@docs
evaluate!
```

```@docs
getvalue
```

# Stopping Criteria
```@docs
isbreak!
```
