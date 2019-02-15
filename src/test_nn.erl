-module(test_nn).

-export([test_activation/0]).


test_activation() ->
	Synapses = deeplearn:init_synapses(2),
	io:format("~n~nSYNAPSES ~w~n", [Synapses]),
	Neurons = deeplearn:spawn_nn(node(), [2,3,1], tanh, Synapses, [self()]),
	io:format("~n~nNEURONS ~w~n", [Neurons]),
	lists:map(fun(X)->X ! {input, 1.0} end, Synapses),
	Prediction = 1.0,
	receive
		{input, _, Activation, WeightedInput} ->
			Error = (Activation - Prediction) * deeplearn:tanh_prime(WeightedInput),
			io:format("->> Error ~w~n", [Error])
	end.
