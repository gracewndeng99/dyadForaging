using CSV
using DataFrames
using Base.Threads

# Run from scripts_final so rl_fit.jl's existing relative paths work.
cd(@__DIR__)

# Load rl_fit.jl definitions without executing its bottom run block.
rl_fit_path = joinpath(@__DIR__, "rl_fit.jl")
rl_fit_source = read(rl_fit_path, String)
run_block = findfirst("########## run", rl_fit_source)
if run_block === nothing
    error("Could not find the run block marker in rl_fit.jl")
end
include_string(Main, rl_fit_source[1:first(run_block)-1], rl_fit_path)

println("Julia threads: ", Threads.nthreads())

folders = [
    # "expl", 
    # "conf",
    "rep2",
]


# Fixed model components for this batch.
GENRULE = "peppgFull"
CHOICERULE = "econ"
# ALPHARULE = "lrdecay"
# PREDICTIONTYPE = "realPrediction"
SCOREFUNC = "_llh"



alpha_rule_list = [
    "lrdecay", 
    # "lrhist",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # "lrflat"
]
prediction_type_list = [
    # "realPrediction", 
    # "rollingAverage", 
    "learned"
]

 
mymodel2_list = [
    # "asIfIdv$(SCOREFUNC)",
    # "updateTheta$(SCOREFUNC)",
    "arbWeight$(SCOREFUNC)",
]


for folder in folders
    println("\n==============================")
    println("Folder: $folder")
    println("==============================")

    for mymodel2 in mymodel2_list
        for alpha_rule in alpha_rule_list
            for prediction_type in prediction_type_list
                global ALPHARULE = alpha_rule
                global PREDICTIONTYPE = prediction_type
                mymodel = "$(PREDICTIONTYPE)_$(ALPHARULE)_$(GENRULE)_$(CHOICERULE)_ThetaGamma"

                println("\n------------------------------")
                println("Running: $(mymodel)_$(mymodel2)_$(folder)")
                println("------------------------------")

                main(
                    folder,
                    mymodel2;
                    recovery=false,
                    mysimtype="full"
                )

                println("Finished: $(mymodel)_$(mymodel2)_$(folder)")
            end
        end
    end
end
