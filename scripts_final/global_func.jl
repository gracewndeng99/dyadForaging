using Pkg
using Printf
# Global parameter
function get_attack_prob(x, max_attack_prob)
    """
    Get the predator's attack probability given the location that the players choose
    """
    if x > 2
        return (x / 20)^2 * max_attack_prob
    else
        return 0.0
    end
end

function get_potential_reward(x)
    """
    Get potential reward of a location
    """
    return x^2
end

function get_predator_choice(x, ptype)
    """
    Determine if predator attacks given the location that the players choose
    """
    if ptype == 1
        prob = get_attack_prob(x, 4.8)
    elseif ptype == 0
        prob = get_attack_prob(x, 2.6)
    end
    return rand(Categorical([1-prob, prob])) - 1 #covert to 0, 1
end

function get_final_step(player, partner)
    """
    Get potential reward of a location
    """
    final = ceil((player + partner) / 2)
    return Int(final)
end


function float_extract(str::String)
    # Remove brackets and any unwanted characters, then replace newlines/spaces with a single space
    cleaned_str = replace(str, r"[\[\]]" => "")  # Remove brackets
    cleaned_str = replace(cleaned_str, r"\s+" => " ")  # Replace multiple spaces/newlines with a single space
    cleaned_str = strip(cleaned_str)  # Trim leading/trailing whitespace

    # Split by spaces and parse to float
    num_strings = split(cleaned_str)  # Split by spaces
    return [parse(Float64, s) for s in num_strings if !isempty(s)]  # Convert each string to Float64
end




