-module(deeplearn).
-export([synapse_loop/1, init_synapses/1, sigmoid/1, sigmoid_prime/1,
		 tanh/1, tanh_prime/1, spawn_nn/5, init_neuron_loop/3, neuron_loop/5]).


% Activation functions
sigmoid(X) -> 1 / (1 + math:exp(-X)).
sigmoid_prime(X) -> sigmoid(X) * (1 - sigmoid(X)).

tanh(X) -> 2 * sigmoid(2 * X) - 1.
tanh_prime(X) -> 1 - tanh(X) * tanh(X).


activate(Weights, Values, Activation) ->
	Val = lists:foldl(
		fun(X, Y) -> X + Y end,
		0,
		lists:zipwith(fun(X, Y) -> X * Y end, [1.0 | Values], Weights)),
	case Activation of
		sigmoid -> sigmoid(Val);
		tanh -> tanh(Val)
	end.

% Activation loop of a neuron waiting for all input values to be received
activation_loop(InNeuron, Value, InNeurons) ->
	lists:map(
		fun(OtherInNeuron) ->
			case OtherInNeuron of 
				InNeuron -> Value;
				_ ->
					receive
						{input, OtherInNeuron, OtherValue} ->
							io:format("[~p] Received input ~w from ~p!~n", [self(), OtherValue, OtherInNeuron]),
							OtherValue;
						stop_input -> 
							io:format("Stoping activation loop~n", []),
							stop
						% X ->
						% 	io:format("!![~p] RECEIVED WIERD ~w, waiting from: ~p ~n", [self(), X, OtherInNeuron])
					% after
					% 	2500 ->
					% 		io:format("[~p] noone sent anything, mailbox: ~w~n", [self(), c:flush()])
					end
			end
		end,
		InNeurons).


synapse_loop(OutNeurons) ->
	receive
		{out, NewOutNeurons} ->
			io:format("[~p] New synapse output ~w ~n", [self(), NewOutNeurons]),
			synapse_loop(NewOutNeurons);
		{input, Value} ->
			io:format("[~p] New synapse input ~w ~n", [self(), Value]),
			lists:map(
				fun(X) -> X ! {input, self(), Value} end,
				OutNeurons),
			synapse_loop(OutNeurons);
		stop_input ->
			io:format("Force activation stop ~n", []),
			lists:map(
				fun(X) -> X ! stop_input end,
				OutNeurons),
			synapse_loop(OutNeurons);
		stop -> ok
	end.

init_synapses(N) ->
	init_synapses(N, []).

init_synapses(0, Synapses) ->
	lists:reverse(Synapses);
init_synapses(N, Synapses) ->
	init_synapses(
		N - 1,
		[spawn(deeplearn, synapse_loop, [[]]) | Synapses]).


neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation) ->
	% check whether monitor node is alive
	case net_adm:ping(Monitor) of
		pong -> 
			ok;
		pang ->
			io:format("CANNOT CONNECT TO THE MONITOR: ~p!~n", [Monitor]),
			neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation)
	end,
	receive
		{input, InNeuron, Value} -> 
			io:format("[~p] Received input ~w from ~p!~n", [self(), Value, InNeuron]),
			Values = activation_loop(InNeuron, Value, InNeurons),
			case lists:member(stop, Values) of
				false -> 
					ActivationValue = activate(Weights, Values, Activation),
					io:format("[~p] Activated successfully with ~p!~n", [self(), ActivationValue]),
					lists:map(
						fun(X) -> X ! {input, self(), ActivationValue} end,
						OutNeurons);
					% io:format("[~p] will send to ~w~n", [self(), OutNeurons]);
				_ -> 
					io:format("Activation of ~p failed!~n", [self()]),
					not_activated
			end,
			neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation);
		{ update, OutNeuron, Value } -> 
			io:format("Received update ~w from ~p!~n", [Value, OutNeuron])
	end,
	neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation).


init_neuron_loop(Monitor, InNeurons, Activation) -> 
	receive
		{out, OutNeurons} -> 
			Weights = 
				[0.01 | [
				rand:normal(0,1) *
				math:sqrt(2/length(InNeurons)) || _ <- lists:seq(1, length(InNeurons)) ]],
			neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation)
	end.

init_layer(_, _, 0, NeuronPids, _, _) ->
	NeuronPids;
init_layer(Node, LayerSize, N, NeuronPids, Activation, PrevNeurons) ->
	Pid = spawn(Node, deeplearn, init_neuron_loop, [Node, PrevNeurons, Activation]),
	init_layer(Node, LayerSize, N-1, [Pid | NeuronPids], Activation, PrevNeurons).

connect_out_neurons([_]) -> ok;
connect_out_neurons([Layer, InLayer | Tail]) ->
	lists:map(fun(N) -> N ! {out, Layer} end, InLayer),
	connect_out_neurons([InLayer | Tail]).


spawn_nn_helper(_, [], _, _, OutNeurons, [LastLayer | PrevNeurons]) -> 
	lists:map(fun(N) -> N ! {in, OutNeurons} end, LastLayer),
	connect_out_neurons([OutNeurons, LastLayer | PrevNeurons]),
	lists:reverse([LastLayer | PrevNeurons]);
spawn_nn_helper(Node, [H | TailLayers], Activation, InputNeurons, OutNeurons, []) ->
	NeuronPids = lists:reverse(init_layer(Node, H, H, [], Activation, InputNeurons)),
	lists:map(fun(N) -> N ! {out, NeuronPids} end, InputNeurons),
	spawn_nn_helper(
		Node, TailLayers, Activation,
		InputNeurons, OutNeurons, 
		[NeuronPids]);
spawn_nn_helper(Node, [H | TailLayers], Activation, InputNeurons, OutNeurons,
			    [PrevNeurons | NeuronsBefore]) ->
	NeuronPids = lists:reverse(init_layer(Node, H, H, [], Activation, PrevNeurons)),
	spawn_nn_helper(Node, TailLayers, Activation,
					InputNeurons, OutNeurons,
					[NeuronPids, PrevNeurons | NeuronsBefore]).

spawn_nn(Node, Layers, Activation, InputNeurons, OutNeurons) ->
	io:format("Spawning a neural network on ~p with layers ~w....~n", [Node, Layers]),
	spawn_nn_helper(
		Node, Layers, Activation,
		InputNeurons, OutNeurons, []).

