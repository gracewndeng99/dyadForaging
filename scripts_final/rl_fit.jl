using CSV
using DataFrames
using Optim
using Distributions
using Random
using Base.Threads
using StatsBase
using JSON

include("global_func.jl")

# define global params
NSTEPS = 9 #9 STEPS to chose from between safety and danger
#max_attack_prob = 4.8 #max attack probability of predator (decreasing with distance)
#opts = np.array([0, 0.25, 0.5, 0.75, 1]) #split options (keep 0 to keep 1)
NT =30 #num trials per predator ==30
REWARDS = (1:NSTEPS) .^2 #rewards of each location
beta = 5

# Define MYMODEL
GENRULE = "peppgFull"
CHOICERULE = "econ"
ALPHARULE = "lrflat" #"lrhist" #"flat" #"lrdecay2"
PREDICTIONTYPE = "rollingAverage"#"realPrediction"#"rollingAverage"#"learned"
SCOREFUNC = "_llh" #"mse" or _llh

# Print the results
println([GENRULE, CHOICERULE])

# Read data
function read_data(folder; prediction_type = PREDICTIONTYPE)
    df_group = CSV.read("../processed_data/parsed_group$(folder).csv", DataFrame)
    df_idv = CSV.read("../processed_data/parsed_idv$(folder).csv", DataFrame)
    #sort by trial!
    # df_idv.trial = parse.(Int, df_idv.trial)
    # df_group.trial = parse.(Int, df_group.trial)
    sort!(df_idv, [:trial])
    sort!(df_group, [:trial])
    
    #add encounter
    transform!(groupby(df_idv, [:subID, :predatorType]), :trial => (x -> 1:length(x)) => :encounter)
    #convert attack from boolean to int
    df_idv.attack = Int.(df_idv.attack)
    # df_group.attack = Int.(df_group.attack)
    df_group.attack = Int.(coalesce.(df_group.attack, 0))

    if prediction_type=="learned"
        # if use learned prediciton, replace real prediction with sim prediction
        pred_sim = CSV.read("../model_fits/rl$(folder)/simulated_predictions$(folder).csv", DataFrame)
        select!(df_group, Not(:prediction))  # drop old prediction column
        df_group = leftjoin(df_group, pred_sim[!, [:subID, :trial, :predatorType, :prediction_argmax, :sigma_used]], 
                            on=[:subID, :trial, :predatorType])
        rename!(df_group, :prediction_argmax => :prediction)
        rename!(df_group, :sigma_used => :sigma)
    else
        df_group[!, :sigma] = fill(1, nrow(df_group))    # use this if the column doesn't exist      
    end


    # add subID
    # subs = unique(df_group[!, :sub])
    # id_mapping = Dict(sub => i for (i, sub) in enumerate(subs))
    # df_group[!, :subID] = [id_mapping[sub] for sub in df_group[!, :sub]]
    # df_idv = filter(row -> row[:sub] in subs, df_idv)
    # df_idv[!, :subID] = [id_mapping[sub] for sub in df_idv[!, :sub]]

    # Identify outliers
    # mean_df = combine(groupby(df_idv, [:subID, :predatorType]), :choice => mean => :mean)
    # df_idv = leftjoin(df_idv, mean_df, on = [:subID, :predatorType])
    # df_idv.outlier = abs.(df_idv.choice .- df_idv.mean) .>= 5

    # filter out weird choices
    df_idv = filter(row -> row[:choice] > 0, df_idv)
    df_group = filter(row -> row[:playerStep] > 0, df_group)

    return df_idv, df_group
end

function get_curr_learning_rate(alpha, curr_trial; alpharule=ALPHARULE)
    if occursin("lrhist", alpharule)
        alpha_t = alpha * (1 - curr_trial/90) #encounter each predator 90 trials
    elseif occursin("lrdecay", alpharule)
        # lrdecay: 
        # alpha_t =  alpha / curr_trial
        # lrdecay2: 
        alpha_t =  alpha / sqrt(curr_trial) #decay with number of encounter
        # lrdecay3: 
        # alpha_t =  alpha / (1 + 0.1 * curr_trial)
    else
        alpha_t = alpha
    end
    return alpha_t
end

function update_V_safety(self_params, history, V_safety;
    gen=GENRULE, phase=1)

    curr_loc = Int(history[end][1])  # Get the current location (1-based index in Julia)
    attack = Int(history[end][2])

    # println(self_params)
    alpha, _, gamma = self_params
    # gamma = 1

    if length(alpha)>1
        alpha_w, alpha_l = alpha
    else
        alpha_w = alpha
        alpha_l = alpha
    end

    if length(gamma)>1
        gamma_f, gamma_b = gamma
    else
        gamma_f = gamma 
        gamma_b = gamma 
    end

    if phase==1
        curr_trial = length(history)
    else
        curr_trial =  60 + length(history)
    end


    if attack == 1# If attacked
        pe = 0 - V_safety[curr_loc]
        #changed here: surprise based lr
        # if split(mymodel, "_")[1] == "lrpe"
        #     alpha_lt = alpha_l * (1 / (1 + abs(pe)))
        # elseif split(mymodel, "_")[1] == "lrhist" #changed here for _new
        #     alpha_lt = alpha_l * (1 - curr_trial/90)
        # end
        # alpha_lt = alpha_l * (1 - curr_trial/90)
        alpha_lt = get_curr_learning_rate(alpha_l, curr_trial)

        V_safety[curr_loc] += alpha_lt * pe # Update on that square

        if curr_loc != NSTEPS # Update all state values after that square

            min_s = curr_loc + 1
            if gen == "valueppg"
                discounted_v = V_safety[curr_loc] .* gamma_f .^ collect(1:(NSTEPS - curr_loc)) #index
                V_safety[min_s:end] = V_safety[min_s:end] .+ alpha_lt .* min.(discounted_v .- V_safety[min_s:end], 0)
            elseif gen == "valueppgFull"
                    discounted_v = V_safety[curr_loc] .* gamma_f .^ collect(1:(NSTEPS - curr_loc)) #index
                    V_safety[min_s:end] = V_safety[min_s:end] .+ alpha_lt .* min.(discounted_v .- V_safety[min_s:end], 0)
                    discounted_v = V_safety[curr_loc] .* (1/gamma_b) .^ reverse(collect(1:curr_loc-1)) #index
                    V_safety[1:curr_loc-1] = V_safety[1:curr_loc-1] .+ alpha_lt .* min.(discounted_v .- V_safety[1:curr_loc-1], 0)
            elseif gen == "peppg"
                discounted_pe = pe .* gamma_f .^ collect(1:(NSTEPS - curr_loc))
                V_safety[min_s:end] = V_safety[min_s:end] .+ alpha_lt * discounted_pe
                # V_safety[min_s:end] = V_safety[min_s:end] .+ gamma * alpha_lt * pe
            elseif gen == "peppgFull"
                discounted_pe = pe .* gamma_f .^ collect(1:(NSTEPS  - curr_loc))
                V_safety[min_s:end] = V_safety[min_s:end] .+ alpha_lt * discounted_pe
                # V_safety[min_s:end] = V_safety[min_s:end] .+ gamma * alpha_lt * pe
                discounted_pe = pe .* gamma_b .^ reverse(collect(1:curr_loc-1))
                V_safety[1:curr_loc-1] = V_safety[1:curr_loc-1] .+ alpha_lt * discounted_pe
            elseif gen == "indlrn"
                gammas = gamma_f .* collect(1:(NSTEPS - curr_loc))
                V_safety[min_s:end] += alpha_lt .* gammas .* (0 .- V_safety[min_s:end])
            elseif gen == "indlrnFull"
                gammas = gamma_f .* collect(1:(NSTEPS - curr_loc))
                V_safety[min_s:end] += alpha_lt .* gammas .* (0 .- V_safety[min_s:end])
                gammas = gamma_b.* reverse(collect(1:curr_loc-1))
                V_safety[1:curr_loc-1] += alpha_lt .* gammas .* (0 .- V_safety[1:curr_loc-1])
            elseif gen=="noGen"
                V_safety = V_safety #skip generalization if no gen
            else
                println("error: please input a valid gen rule")
            end
        end

    else # Not attacked
        pe = 1 - V_safety[curr_loc]
        #changed here: surprise based lr
        # if split(mymodel, "_")[1] == "lrpe"
        #     alpha_wt = alpha_w * (1 / (1 + abs(pe)))
        # elseif split(mymodel, "_")[1] == "lrhist"
        #     alpha_wt = alpha_w * (1 - curr_trial/90)
        # end
        # alpha_wt = alpha_w * (1 - curr_trial/90)
        alpha_wt = get_curr_learning_rate(alpha_w, curr_trial)  
        
        V_safety[curr_loc] += alpha_wt * pe # Update on that square
        if curr_loc != 1 # Update all state values before that square

            max_s = curr_loc - 1
            if gen == "valueppg"
                discounted_v = V_safety[curr_loc] .* (1/gamma_b) .^ reverse(collect(1 : max_s)) #index
                # println(max.(discounted_v .- V_safety[1:max_s], 0))
                V_safety[1:max_s] = V_safety[1:max_s] .+ alpha_wt .* max.(discounted_v .- V_safety[1:max_s], 0)
            elseif gen == "valueppgFull"
                discounted_v = V_safety[curr_loc] .* (1/gamma_b) .^ reverse(collect(1 : max_s)) #index
                # println(max.(discounted_v .- V_safety[1:max_s], 0))
                V_safety[1:max_s] = V_safety[1:max_s] .+ alpha_wt .* max.(discounted_v .- V_safety[1:max_s], 0)
                discounted_v = V_safety[curr_loc] .* gamma_f .^ collect(1:NSTEPS-curr_loc)
                V_safety[curr_loc+1:NSTEPS] = V_safety[curr_loc+1:NSTEPS] .+ alpha_wt .* max.(discounted_v .- V_safety[curr_loc+1:NSTEPS], 0)
            elseif gen == "peppg"
                discounted_pe = pe .* gamma_b .^ reverse(collect(1 : max_s))
                V_safety[1:max_s] .= V_safety[1:max_s] .+ alpha_wt * discounted_pe
            elseif gen == "peppgFull"
                discounted_pe = pe .* gamma_b .^ reverse(collect(1 : max_s))
                V_safety[1:max_s] .= V_safety[1:max_s] .+ alpha_wt * discounted_pe
                # V_safety[1:max_s] .= V_safety[1:max_s] .+ gamma * alpha_wt * pe
                discounted_pe = pe .* gamma_f .^ collect(1:NSTEPS-curr_loc)
                V_safety[curr_loc+1:NSTEPS] = V_safety[curr_loc+1:NSTEPS] .+ alpha_wt * discounted_pe
            elseif gen == "indlrn"
                gammas = gamma_b .* reverse(collect(1 : max_s))
                V_safety[1:max_s] += alpha_wt .* gammas .* (1 .- V_safety[1:max_s])
            elseif gen == "indlrnFull"
                gammas = gamma_b .* reverse(collect(1 : max_s))
                V_safety[1:max_s] += alpha_wt .* gammas .* (1 .- V_safety[1:max_s])
                gammas = gamma_f .* collect(1:NSTEPS-curr_loc)
                V_safety[curr_loc+1:NSTEPS] += alpha_wt .* gammas .* (1 .- V_safety[curr_loc+1:NSTEPS])
            elseif gen=="noGen"
                V_safety = V_safety
            else
                println("error: please input a valid gen rule")
            end
        end
    end

    # Clip the values between 0 and 1
    V_safety .= clamp.(V_safety, 0, 1)
    return V_safety
end


# get the value for each specified location
function get_location_util(loc, V_safety, params; 
    choice_rule=CHOICERULE, rewards=REWARDS)
    # Modified from 2_2_modeling_idv_agents_wo_learning
    _, theta, _ = params  # Access the theta

    i = Int(loc);
    if choice_rule == "surv"
        normalized_rewards = (rewards .- minimum(rewards)) ./ (maximum(rewards) - minimum(rewards))
        Q = theta * V_safety[i] + (1 - theta) * normalized_rewards[i]  # +1 for 1-based indexing
    elseif choice_rule == "econ"
        Q = V_safety[i] * rewards[i] ^ theta - (1 - V_safety[i]) * 10 ^ theta
    elseif choice_rule == "prelec"
        V_safety_t = exp.(-(-log(V_safety[i])) .^ theta)  # Element-wise exponentiation
        Q = V_safety_t * rewards[i] - (1 - V_safety_t) * 10
    else
        println("Input a valid choice rule")
        return NaN  # Return NaN for invalid input
    end

    return Q
end


# get self preference based on V_safety and self params: normalized to 0-1
function get_self_pref(V_safety, idv_params)
    """
    Utility for locations for player 1.
    Compensate for partner's behavior to maximize self-reward.
    Takes partner_step as predicted partner location.
    """
    # #if updateTheta2:
    # utils = [get_location_util(ceil((i + pred_partner_step) / 2), V_safety, idv_params) for i in 1:NSTEPS]
    # idv_params[end] = idv_params[end] + delta
    #if updateTheta:
    utils = [get_location_util(i, V_safety, idv_params) for i in 1:NSTEPS]
    
    # Normalize utilities to the same scale as partner preferences
    min_util = minimum(utils)
    max_util = maximum(utils)

    den = max_util - min_util
    normalized_utils = den > 1e-12 ? (utils .- min_util) ./ den : fill(0.0, length(utils))
    
    return normalized_utils
end



# get sub llh
function get_sub_llh_idv(params, subdf::DataFrame; scorefunc=SCOREFUNC, beta=beta)

    # #constraints
    # for i in 1:length(params)
    #     if (params[i] < bounds[i][1]) || (params[i] > bounds[i][2])
    #         return 1e100
    #     end
    # end

    score = 0  # Initialize an empty list to store likelihood values
    for pt in [0, 1]
        g = sort!(subdf[subdf.predatorType .== pt, :], :trial)
        # println(g[1:5, :])
        # Initialize V_safety
        V_safety = collect(1.0 : -1/NSTEPS : 1/NSTEPS)
        # Initialize history as a list of tuples
        history = [Tuple(g[1, [:choice, :attack]])]
        
        for row in eachrow(g)  # Start from round 2
            # println(row.encounter)
            # println(V_safety)
            if row.encounter!=1
                #update V_safety
                V_safety = update_V_safety(params, history, V_safety)
                #calculate location value
                loc_v = get_self_pref(V_safety, params)
                # Get likelihood of subject's current choice
                #if row.outlier ==false #only record none outlier trial
                # llh_t = occursin("mse", mymodel) ? (row.choice - argmax(choice_prob)) ^ 2 : -log(choice_prob[row.choice])  # Add 1 for 1-indexing in Julia
                # mse_t = (row.choice - argmax(loc_v)) ^ 2
                # mse += mse_t
                # if use likelihood
                if occursin("llh", scorefunc)
                    choice_prob = exp.(beta .* loc_v)
                    choice_prob ./= sum(choice_prob)
                    score_t = -log(choice_prob[row.choice])
                else
                    # choice_llh[prev_loc] += bias
                    score_t = (row.choice - argmax(loc_v)) ^ 2
                end
                
                score += score_t
                #push to history
                push!(history, (row.choice, row.attack))  # Append new tuple to history
            end
            
        end
    end
    return score  # Sum of negative log-likelihood
end

# function to get V_safety given subject's real choice
function get_Vsafety(bf_params, subdf)
    #mytype = full, partial
    #initialize as array of array
    Vsafety_list = Vector{Vector{Float64}}()
    for pt in [0, 1]
        g = sort(subdf[subdf.predatorType .== pt, :], :trial) #get predator df
        V_safety = collect(1.0 : -1/NSTEPS : 1/NSTEPS)  # Initialize V_safety
        history = [Tuple(g[1, [:choice, :attack]])]  # Initialize history as a list of tuples
        
        for row in eachrow(g)  # Start from round 2
            if row.encounter!=1
                V_safety = update_V_safety(bf_params, history, V_safety)
                # choice_prob = get_choice_value_m1(bf_params, V_safety, history)
                # Append new tuple to history
                push!(history, (row.choice, row.attack)) 
            end
        end
        push!(Vsafety_list, V_safety)
    end

    return Vsafety_list
end

# function to simulate choices
function sim_choice_idv(bf_params, subdf; mytype="partial", beta=beta)
    #mytype = full, partial
    #initialize as array of array
    # Vsafety_list = Vector{Vector{Float64}}()
    sim_choices = Dict{Int, Int}()
    sim_attacks = Dict{Int, Int}()
    for pt in [0, 1]
        g = sort(subdf[subdf.predatorType .== pt, :], :trial)
        # Initialize V_safety and history
        history =[]
        V_safety = []
        # history = [Tuple(g[1, [:choice, :attack]])]  # Initialize history as a list of tuples

        for row in eachrow(g)  
            if row.encounter==1
                sim_choices[row.trial] = row.choice
                sim_attacks[row.trial] = row.attack
                push!(history, (row.choice, row.attack))
                V_safety = collect(1.0 : -1/NSTEPS : 1/NSTEPS)
                # println(V_safety)
            else
                V_safety = update_V_safety(bf_params, history, V_safety)
                loc_v = get_self_pref(V_safety, bf_params)
                #get choice prob
                choice_prob = exp.(loc_v .* beta) ./ sum(exp.(loc_v .* beta))
                # try
                #simulate a choice
                sim_choice = rand(Categorical(choice_prob))
                #save simulated choice
                sim_choices[row.trial] = sim_choice
                # catch e
                #     sim_choice = rand(1:9)
                #     #save simulated choice
                #     sim_choices[row.trial] = sim_choice
                # end

                if mytype=="partial"
                    # Append what actually happened to history
                    push!(history, (row.choice, row.attack))  
                    sim_attacks[row.trial] = row.attack
                elseif mytype=="full"
                    #simulate a attack
                    sim_attack = get_predator_choice(sim_choice, pt)
                    # Append simulation to to history
                    push!(history, (sim_choice, sim_attack)) 
                    sim_attacks[row.trial] = sim_attack
                end
            end
        end
        # push!(Vsafety_list, V_safety)
    end
    # println("Simulated choices: ", sim_choices, "Simulated attacks: ", sim_attacks)
    return sim_choices, sim_attacks
end


# function to get V_safety given subject's real choice
function sim_sub_idv(bf_params_df, df_idv; mytype="partial")
    sim_df_idv = DataFrame(subID = Int[], trial=Int[], sim_choice = Int[], sim_attack = Int[])
    for s in unique(bf_params_df.subID)
        # for s in [203,204]
        subdf = df_idv[df_idv.subID .== s, :]
        # Get self parameters
        bf_params = bf_params_df[bf_params_df.subID .== s, [:alpha, :theta, :gamma]]
        bf_params = vec(Matrix(bf_params))
        println("Sub: $s, Params: $bf_params")
        
        #sim
        # try #in case some subs are excluded in first phase
        sim_choices, sim_attacks = sim_choice_idv(bf_params, subdf, mytype=mytype)
        sim_df = DataFrame(trial=collect(keys(sim_choices)), sim_choice=collect(values(sim_choices)), sim_attack = collect(values(sim_attacks)))
        sim_df.subID .= s
        #merge two dfs. 
        sim_df_idv = vcat(sim_df_idv, sim_df) 
        # catch e
        #     println("Failed to simulate sub $s")
        # end
    end
    return sim_df_idv
end


# function opt(sub_df, bounds; iters=30)

#     best_v = Inf  # Initialize best value to infinity
#     best_params = fill(NaN, length(bounds))  # Initialize best parameters with NaN

#     for _ in 1:iters
#         # Generate random start values within bounds
#         start = [rand() for i in 1:length(bounds)]
#         # Minimize using the `optimize` function
#         #### if call nlh2, also comment out the V_safety line in get_choice_value_m1
#         res = optimize(params -> get_sum_nlh(params, bounds, sub_df), start)

#         if res.minimum < best_v
#             best_v = res.minimum  # Update best value
#             best_params = res.minimizer  # Update best parameters
#         end
#     end

#     return best_params, best_v
# end




function get_choice_value_m2(weight, idv_params, V_safety, other_pref, mymodel2;
    show_progress=false)
    # pred_partner_step: the step where the partner is most likely to act, based on other_pref

    # self_pref: preference of self after compensating for partner's choice
    self_pref = get_self_pref(V_safety, idv_params) 
    
    # # Combining self and other preferences based on the weight
    # loc_v = weight * other_pref + (1 - weight) * self_pref
    if occursin("socReward", mymodel2)
        pred_partner_step = argmax(other_pref) 
        # self_other_diff = [abs(pred_partner_step - i) for i in range(1, NSTEPS)]
        # loc_v = self_pref .- weight .* self_other_diff
        positions = collect(1:NSTEPS)
        if argmax(self_pref) < pred_partner_step
            loc_v = self_pref .+ weight .* (positions .- pred_partner_step)
        else
            loc_v = self_pref .- weight .* (positions .- pred_partner_step)
        end

    elseif occursin("arbWeight", mymodel2)
        
        # new: Combining self and other preferences based on the weight
        if weight>=0
            loc_v = weight .* other_pref .+ (1 - weight) .* self_pref
        else
            # if using realprediction
            # compensate = clamp(2 * argmax(self_pref) - pred_partner_step, 1, NSTEPS)
            # Q_compensate = exp.(-(collect(1:NSTEPS) .- compensate).^2 ./ (2 * 1^2))
            # # Q_compensate = [max(1.1 - 0.1 * 2^abs(i - compensate), 0) for i in 1:NSTEPS]
            # # Q_compensate = exp.(abs.(collect(1:NSTEPS) .- compensate))
            # loc_v = -weight .* Q_compensate .+ (1 + weight) .* self_pref
            
            ## if using rolling average or learned
            Q_compensate = zeros(Float64, NSTEPS)
            i_self = argmax(self_pref)
            for j in 1:NSTEPS
                mirrored_j = clamp(2 * i_self - j, 1, NSTEPS)
                Q_compensate[mirrored_j] += other_pref[j]
            end
            loc_v = -weight .* Q_compensate .+ (1 + weight) .* self_pref
        end
    
    else
        loc_v = self_pref
    end

    
    if show_progress
        println("weight = $weight")
        println("self: $self_pref")
        println("other: $other_pref")
        println(loc_v)
    end

    # If the model is based on Mean Squared Error
    # if occursin("mse", mymodel)
    #     return loc_v
    # else
    #     # Exponentiate and normalize to get choice probabilities
    #     choice_prob = exp.(beta * loc_v)
    #     choice_prob = choice_prob / sum(choice_prob)

    #     return choice_prob
    # end


    return loc_v
end



# function get_surprise(outcome, self_choice, other_pref, loc_v, base="util", rewards=REWARDS)
#     #takes: outcome-> tuple; self choice-> value; other_pref-> array; loc_v-> array;
#     #Expected return - actual return
#     curr_loc = outcome[1]
#     attack = outcome[2]

#     if base=="util"
#         pred_partner_step = argmax(other_pref)
#         normalized_rewards = (rewards .- minimum(rewards)) ./ (maximum(rewards) - minimum(rewards))
#         # if !isinstance(pred_partner_choice, int):
#         #     pred_partner_choice = np.where(pred_partner_choice==1)[0]
#         # print([self_choice, pred_partner_choice])
#         expected_loc = Int(ceil((self_choice + pred_partner_step) /2))
        
#         if attack
#             surprise = -1 - loc_v[expected_loc]
#         else
#             surprise = normalized_rewards[curr_loc] - loc_v[expected_loc]
#         end

#     elseif base=="dist"
#             surprise = (2*curr_loc - self_choice)/NSTEPS
#         if attack
#             surprise = -surprise
#         end
#     end


#     return surprise
# end


# function update_weight(w, surprise, delta)
#     new_w = w + delta * surprise
#     return new_w

# end 

function get_sub_llh_grp(params_to_optimize, sub_df, Vsafety_list, mymodel2; 
    gen=GENRULE, choice=CHOICERULE, scorefunc=SCOREFUNC, beta=beta,
    show_progress=false)
    
    # Initialize variables
    all_score = []
    blame=0.5

    #initialize to 0
    weight = 0
    theta_g = 0

    # know which model to run
    if occursin("updateTheta", mymodel2)
        theta_g = params_to_optimize[end]
    elseif occursin("arbWeight", mymodel2) || occursin("socReward", mymodel2)
        weight = params_to_optimize[end]
    end
    
    self_params = [params_to_optimize[1], max(params_to_optimize[2]+theta_g, 0), params_to_optimize[3]]

    # #constraints:
    # if ((self_params1[3]+theta_g) <0) || ((self_params1[3]+theta_g) >2)
    #     return 1e100
    # end
    # if (weight <-1) || (weight>1)
    #     return 1e100
    # end
    # if (delta <-1) || (delta>1)
    #     return 1e100
    # end
    
    for pt in [0, 1]
        g = sort!(sub_df[sub_df.predatorType .== pt, :], :trial)
        # Initialize V_safety and history
        other_history = []
        self_history =[]
        V_safety = []

        # Start the loop over trials
        for row in eachrow(g)
            # println([row.trial, row.subID])
            if row.trial == 1
                # Initialize variables at the first trial
                Q_partner = fill(0.0, NSTEPS) / NSTEPS
                V_safety = Vsafety_list[row.predatorType+1]
                # println(V_safety)
                other_pref = fill(1.0 / NSTEPS, NSTEPS)  # Flat prior
                blame = row.selfBlame != -1 ? row.selfBlame : 0.5

            else
                
                # println(row.trial)
                # println(self_history)
                # Update safety value and other preferences
                V_safety = update_V_safety(self_params, self_history, V_safety, phase=2)
                # pred_other_step = row.prediction != -1 ? row.prediction : other_history[end]
                # Q_partner = [max(1.1 - 0.1 * 2^abs(i - pred_other_step), 0) for i in 1:NSTEPS]
                # Q_partner = get_partner_pref([pred_other_step])
                Q_partner = get_partner_pref(row.prediction, other_history; sigma=row.sigma)
                # Calculate Q
                loc_v = get_choice_value_m2(weight, self_params, V_safety, Q_partner, mymodel2)

                prev_loc = self_history[end][1]

                choice = row.playerStep 
                if (row.step_rt<8) && (choice>0)
                    if occursin("llh", scorefunc)
                        choice_prob = exp.(beta .* loc_v)
                        choice_prob ./= sum(choice_prob)
                        score_t = -log(choice_prob[choice])
                    else
                        # choice_llh[prev_loc] += bias
                        score_t = (choice - argmax(loc_v)) ^ 2
                    end
                    # Store choice log likelihood if the choice is not computer generated
                    push!(all_score, score_t)
                end

                if show_progress
                    println("P(choice = $(row.playerStep)) = $choice_llh on trial $(row.trial)")
                end

                # if isnan(score_t)
                #     println("NaN score at trial $(row.trial) for sub $(row.subID)")
                #     println([loc_v, choice_prob])
                # end

                blame = row.selfBlame != -1 ? row.selfBlame : 0.5
            end

            # Update histories
            push!(other_history, row.partnerStep)
            push!(self_history, (row.finalStep, row.attack))
            # push!(surprise_list, surprise)
            # push!(weight_list, weight)
    
            # Record updated safety at the end
            # if row.trial == 30
            #     safety_list[row.predatorType] = V_safety
            # end
        end
    end

    return sum(all_score)
end


# function to get partner preference
function get_partner_pref(real_prediction, partner_history; 
    prediction_type=PREDICTIONTYPE, sigma = 1)

    # Simple preference model: higher preference for locations closer to predicted partner step
    if prediction_type == "realPrediction"
        pred_partner_step = real_prediction != -1 ? real_prediction : partner_history[end]
        partner_pref = exp.(-(collect(1:NSTEPS) .- pred_partner_step).^2 ./ (2 * sigma^2)) #kernal
        # partner_pref = exp.(-abs.(collect(1:NSTEPS) .- pred_partner_step)) # 
        # partner_pref = [max(1.1 - 0.1 * 2^abs(i - pred_partner_step), 0) for i in 1:NSTEPS] # this is not smooth
    
    ## use average of partner history as prediction
    elseif prediction_type == "rollingAverage"
        if isempty(partner_history)
            return fill(0.5, NSTEPS)
        end

        weights = (1 ./ (length(partner_history):-1:1))  # Weights for the last N choices
        weights = weights / sum(weights)  # Normalize weights to sum to 1

        partner_pref = zeros(Float64, NSTEPS)
        NSTEPS_array = collect(1:NSTEPS)
        
        for (c, w) in zip(partner_history, weights)
            partner_pref .+= w .* exp.(-(( NSTEPS_array .- c).^2) ./ (2 * sigma^2))
        end

    ##  read from learning
    elseif prediction_type == "learned"
        mu = real_prediction
        partner_pref = exp.(-((collect(1:NSTEPS) .- mu) .^ 2) ./ (2 * sigma^2))
    end
    # vmax = maximum(partner_pref)
    # vmin = minimum(partner_pref)
    # partner_pref = (partner_pref .- vmin) ./ (vmax - vmin)
    return partner_pref
end


# function to simulate choices
function sim_choice_grp(bf_params, subdf, Vsafety_list, mymodel2; mytype="partial")
    #mytype = full, partial
    #initialize as array of array
    sim_choices = Dict{Int, Dict{Int, Int}}()
    sim_attacks = Dict{Int, Dict{Int, Int}}()
    self_params = bf_params[1:3]
    
    weight = 0
    if occursin("arbWeight", mymodel2) || occursin("socReward", mymodel2)
        weight = bf_params[end]
    elseif occursin("updateTheta", mymodel2)
        # self_params[3] = bf_params[3] + bf_params[4]
        self_params[2] = max(bf_params[2] + bf_params[4], 0)
    end

    # Vsafety_list = get_Vsafety(par[1:3], sub_df_idv)
    for pt in [0, 1]
        g= sort(subdf[subdf.predatorType .== pt, :], :trial)
        self_history=[]
        other_history = []
        V_safety = []

        for row in eachrow(g)  
            if row.trial==1
                Q_partner = fill(0.0, NSTEPS) / NSTEPS
                push!(other_history, row.partnerStep)
                push!(self_history, (row.finalStep, row.attack))
                V_safety = Vsafety_list[pt+1]
                sim_choices[pt] = Dict{Int, Int}()
                sim_choices[pt][row.trial] = row.playerStep
                sim_attacks[pt] = Dict{Int, Int}()
                sim_attacks[pt][row.trial] = row.attack

            else # Start from round 2
                V_safety = update_V_safety(self_params, self_history, V_safety, phase=2)
                # pred_other_step = row.prediction != -1 ? row.prediction : other_history[end]
                # Q_partner = [max(1.1 - 0.1 * 2^abs(i - pred_other_step), 0) for i in 1:NSTEPS]
                # curr_trial_pred= filter(r -> r.subID == row.subID && r.predatorType == row.predatorType && r.trial == row.trial, learned_partner_pred)[1]

                Q_partner = get_partner_pref(row.prediction, other_history)
                
                loc_v = get_choice_value_m2(weight, self_params, V_safety, Q_partner, mymodel2)
                #get choice prob
                choice_prob = exp.(loc_v .* beta) ./ sum(exp.(loc_v .* beta))
                ##simulate a choice and save
                sim_choice = rand(Categorical(choice_prob))
                sim_choices[pt][row.trial] = sim_choice
                

                #append to other history
                push!(other_history, row.partnerStep)
                if mytype=="partial"
                    # Append what actually happened to history
                    push!(self_history, (row.finalStep, row.attack))  
                    sim_attacks[pt][row.trial] = row.attack
                elseif mytype=="full"
                    #final step using sim choice
                    final_loc = get_final_step(sim_choice, row.partnerStep)
                    #simulate a attack
                    sim_attack = get_predator_choice(final_loc, pt)
                    # Append simulation to to history
                    push!(self_history, (final_loc, sim_attack)) 
                    sim_attacks[pt][row.trial] = sim_attack
                end                
                
            end

            # println(V_safety)
        end
    end
    return sim_choices, sim_attacks
end


# function to get simulate group choice
function sim_sub_grp(bf_params_df, df_idv, df_grp, mymodel2; mytype="partial")
    sim_df_grp = DataFrame(subID = Int[], trial=Int[], predatorType=Int[], sim_playerStep=Int[], sim_attack=Int[])
    for s in unique(bf_params_df.subID)
        # Get self parameters
        bf_params = bf_params_df[bf_params_df.subID .== s, [:alpha, :theta, :gamma, :w]]
        bf_params = vec(Matrix(bf_params))
        println("Sub: $s, Params: $bf_params")

        #get df
        sub_df_idv = df_idv[df_idv.subID .== s, :]
        sub_df_grp = df_grp[df_grp.subID .== s, :]
        
        #get V_safety
        Vsafety_list = get_Vsafety(bf_params[1:3], sub_df_idv)
    
        # try #in case some subs are excluded in first phase
        sim_choices, sim_attacks = sim_choice_grp(bf_params, sub_df_grp, Vsafety_list, mymodel2, mytype=mytype)
        # print(sim_attacks)
        # Convert to DataFrame
        flattened_data = [(pt, trial, sim_choice) for (pt, trials) in sim_choices for (trial, sim_choice) in trials]
        sub_sim_df = DataFrame(flattened_data, [:predatorType, :trial, :sim_playerStep])
        flattened_data2 = [(pt, trial, sim_attack) for (pt, trials) in sim_attacks for (trial, sim_attack) in trials]
        sub_sim_df2 = DataFrame(flattened_data2, [:predatorType, :trial, :sim_attack])

        sub_sim_df = innerjoin(sub_sim_df, sub_sim_df2, on=[:predatorType, :trial])
        sub_sim_df.subID .= s

        #append to full dataframe
        sim_df_grp = vcat(sim_df_grp, sub_sim_df) 
        # catch e
        #     println("Failed to simulate sub $s")
        # end
    end

    return sim_df_grp
end


function grid_search(sub_df_idv, sub_df_grp, bounds, grid_size, mymodel2)
    # generate a list of tuples with all possible param combo
    myvar_list = []
    myprior_list = []
    for i in 1:length(bounds)
        bound = bounds[i]
        # myvar = range(bound[1], stop=bound[2], step=(bound[2] - bound[1])/50)
        myvar = bound[1]:grid_size:bound[2]
        push!(myvar_list, myvar)
        # myprior = [pdf(prior_dists[i], x) for x in myvar]
        # push!(myprior_list, myprior)
    end
    combos = collect(Iterators.product(myvar_list...))

    mse = zeros(length(combos))
    for i in 1:length(combos)
        par = collect(combos[i])
        ## if fix alpha:
        # par[1] = 1
        # println(combos[i])
        #get mse for idv phase
        alh = get_sub_llh_idv(par[1:3], sub_df_idv) 
        # get Vsafety 
        Vsafety_list = get_Vsafety(par[1:3], sub_df_idv)
        alh  = alh + get_sub_llh_grp(par, sub_df_grp, Vsafety_list, mymodel2)
        # alh = get_sum_nlh(par, sub_df)
        # println(alh)
        #likelihood approximately equal to exp(-avg_MSE/2)
        mse[i] = alh
    end

    best_idx = argmin(mse)
    best_v = mse[best_idx]
    best_params = combos[best_idx]
    # println(best_params)
    return best_params, best_v
end


#for efficiency
function generation_grid_search(curr_bounds, curr_grid_size, sub_df_idv, sub_df_grp, step_size, mymodel2; 
            orig_bounds = bounds)
    best_params, best_v = grid_search(sub_df_idv, sub_df_grp, curr_bounds, curr_grid_size, mymodel2)
    new_bounds = [(max(best_params[i] - curr_grid_size/2, orig_bounds[i][1]), 
                    min(best_params[i] + curr_grid_size/2, orig_bounds[i][2])) for i in eachindex(best_params)]
    new_grid_size = curr_grid_size / step_size
    return new_bounds, new_grid_size
end


#main function to fit
function fit_all(df_idv, df_grp, bounds, mymodel2, output_fname; 
    step_size = 5, save=true)

    # Calculate grid size for parameter bounds
    # grid_size = [(maximum(b) - minimum(b)) / 100 for b in bounds]
    # grids = [collect(range(minimum(bounds[i]), stop=maximum(bounds[i]), length=100)) for i in 1:length(bounds)]
    params_df = DataFrame(subID=Int[], alpha = Float64[],  theta = Float64[],  gamma = Float64[],  w = Float64[], nll=Float64[])
    # println(phase1_params)
    #Iterate over unique rooms (skipping the first one)
    Threads.@threads for s in unique(df_idv.subID)
    # for s in [203,204]
    # try
    sub_df_idv = df_idv[df_idv.subID .== s, :]
    sub_df_grp = df_grp[df_grp.subID .== s, :]
    
    # Get self parameters and group data for the two people in the room
    # p1_params = phase1_params[phase1_params.subID .== s, [:alpha, :gamma, :theta]][1, :]
    println("Sub: $s")
    #grid search
    # try #in case some subs are excluded in first phase
    # best_params, best_v = grid_search(sub_df_idv, sub_df_grp, bounds, grid_size, mymodel2)
    new_bounds = bounds
    new_grid_size =  1 / step_size
    for i = 1:2 #run 3 times
        new_bounds, new_grid_size = generation_grid_search(
            new_bounds, new_grid_size, sub_df_idv, sub_df_grp, step_size, mymodel2)
    end
    best_params, best_nll = grid_search(sub_df_idv, sub_df_grp, new_bounds, new_grid_size*5/4, mymodel2)
    #round and log: step size is 0.01 or 0.02
    best_params = [round(i, digits=2) for i in best_params]
    println([best_params, best_nll])
    #save to df
    push!(params_df, (s, best_params[1], best_params[2], best_params[3], best_params[4], best_nll))                               
    # catch e
    #     println("Check sub $s")
    # end

    end #end for loop

    if save
        #save
        CSV.write("$(output_fname).csv", params_df)
        println("Parameters saved to $output_fname")
    end
    return params_df
end


#main function to simulate all
function simulate_all(df_idv, df_grp, bf_params_df, mymodel2, output_path, mytype, k)

    #bf_params_df = best fitting parameters for each subject
    #mytype="partial" or "full" using real vs simulated choice in history
    sim_idv = sim_sub_idv(bf_params_df, df_idv, mytype=mytype)
    sim_idv.k .= 1
    sim_grp = sim_sub_grp(bf_params_df, df_idv, df_grp, mymodel2, mytype=mytype)
    sim_grp.k .= 1
        
    for i in 2:k
        #idividual
        subset = sim_sub_idv(bf_params_df, df_idv, mytype=mytype)
        subset.k .= i
        sim_idv = vcat(sim_idv, subset)

        #group
        subset = sim_sub_grp(bf_params_df, df_idv, df_grp, mymodel2, mytype=mytype)
        subset.k .= i
        sim_grp = vcat(sim_grp, subset)
    end
    
    # compress simulations
    sim_idv = combine(
            groupby(sim_idv, [:trial, :subID]),
            :sim_choice => (x -> [collect(x)]) => :sim_choice,  # list of all sim choices
            :sim_attack => (x -> [collect(x)]) => :sim_attack,  # list of all sim attacks
            :k => first => :k  # keep k (should be same within group)
        )
    sim_grp = combine(
            groupby(sim_grp, [:trial, :predatorType, :subID]),
            :sim_playerStep => (x -> [collect(x)]) => :sim_playerStep,  # list of all sim choices
            :sim_attack => (x -> [collect(x)]) => :sim_attack,
            :k => first => :k  # keep k (should be same within group)
        )

    #save
    CSV.write("$(output_path)_sim_idv_$(mytype).csv", sim_idv)
    CSV.write("$(output_path)_sim_group_$(mytype).csv", sim_grp)
    println("simulated df saved to $output_path")
end

# main function for recovery
function recover_all(input_fname, mymodel2, df_idv, df_grp; sim_type="full")
    # read simulated data
    sim_idv = CSV.read("$(input_fname)_sim_idv_$(sim_type).csv", DataFrame)
    sim_grp = CSV.read("$(input_fname)_sim_group_$(sim_type).csv", DataFrame)
    # sim_idv, sim_grp = simulate_all(df_idv, df_grp, params_df, mymodel2, output_fname, mytype, k)
    # sim_idv.trial = parse.(Int, sim_idv.trial)
    sim_idv = leftjoin(sim_idv, df_idv, on=[:subID, :trial])
    sort!(sim_idv, [:subID, :trial])
    transform!(groupby(sim_idv, [:subID, :predatorType]), :trial => (x -> 1:length(x)) => :encounter)
    # sim_grp.trial = parse.(Int, sim_grp.trial)
    sim_grp = leftjoin(sim_grp, df_grp, on=[:subID, :trial, :predatorType])
    sort!(sim_grp, [:room, :subID, :trial])
    # drop and rename columns
    select!(sim_idv, Not(:choice, :attack))
    select!(sim_grp, Not(:playerStep, :attack, :finalStep))
    rename!(sim_idv, :sim_choice => :choice)
    sim_idv[!, :choice] = JSON.parse.(sim_idv[!, :choice])
    sim_idv[!, :choice] = Int.(first.(sim_idv[!, :choice]))
    rename!(sim_grp, :sim_playerStep => :playerStep)
    sim_grp[!, :playerStep] = JSON.parse.(sim_grp[!, :playerStep])
    sim_grp[!, :playerStep] = Int.(first.(sim_grp[!, :playerStep]))
    #coearce missing value to 5
    sim_grp.partnerStep = Int.(coalesce.(sim_grp.partnerStep, 5))
    transform!(sim_idv, [:choice, :predatorType] => ByRow(get_predator_choice) => :attack)
    transform!(sim_grp, [:playerStep, :partnerStep] => ByRow(get_final_step) => :finalStep)
    transform!(sim_grp, [:finalStep, :predatorType] => ByRow(get_predator_choice) => :attack)
    #fit
    params_df = fit_all(sim_idv, sim_grp, bounds, mymodel2, output_fname; step_size = 5, save=false)
    #save
    CSV.write("$(input_fname)_recovery.csv", params_df)
    println("recovered df saved to $(input_fname)_recovery")
end



# folder = "" 
# folder = "_rep2"
folder = "_conf"

## read data
df_idv, df_grp = read_data(folder)

#define model names
mymodel = "$(PREDICTIONTYPE)_$(ALPHARULE)_$(GENRULE)_$(CHOICERULE)_ThetaGamma"
println("Fitting model: $mymodel")
# mymodel2 = "socReward"
# mymodel2 = "asIfIdv$(SCOREFUNC)"
# mymodel2 = "updateTheta$(SCOREFUNC)"
mymodel2 = "arbWeight$(SCOREFUNC)"
# Define bounds (alpha, theta, gamma, w/gamma2)
bounds = [(0, 1), (0, 1.5), (0, 1), (-1, 1)]
#define outputpath
output_fname = "../model_fits/rl$(folder)/$(mymodel)_$(mymodel2)$(folder)"
# println("Output path: $output_fname")
# # Fit the model and return parameters
params_df = fit_all(df_idv, df_grp, bounds, mymodel2, output_fname)
# params_df.alpha .= 1

# # Simulate data
# mytype = "partial"
mytype = "full"
k = 10
params_df = CSV.read("$(output_fname).csv", DataFrame)
simulate_all(df_idv, df_grp, params_df, mymodel2, output_fname, mytype, k)

# Model recovery
# recover_all(output_fname, mymodel2, df_idv, df_grp)
