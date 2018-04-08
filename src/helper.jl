@inline function maximumbelowInf(values)
    m = -Inf64
    for v in values
        if v < Inf64 && v > m
            m = v
        end
    end
    if m == -Inf64
        Inf64
    else
        m
    end
end

macro subtypes(supertype, body, subtypes...)
    for subtype in subtypes
        @eval (mutable struct $subtype <: $supertype
            $body
        end;
        export $subtype)
    end
end

macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

@def common_learner_fields begin
    γ::Float64
    buffer::Tbuff
end
