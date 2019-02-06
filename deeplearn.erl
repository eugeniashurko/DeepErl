-module(deeplearn).
-export([step/1, sigmoid/1, tanh/1, spawn_nn/3, init_neuron_loop/3]).

% Activation functions
step(X) when X > 0 -> 1;
step(X) when X == 0 -> 0.5;
step(_) -> 0.

sigmoid(X) -> 1 / (1 + math:exp(-X)).

tanh(X) -> 2 * sigmoid(2 * X) - 1.


neuron_loop(MonitorNode, OutNeurons, InNeurons, Activation) ->
	% check whether monitor node is alive
	io:format("Entered ~p neuron loop [activation=~w] with in neurons ~w, out neurons  ~w...~n",
			  [self(), Activation, OutNeurons, InNeurons]),
	case net_adm:ping(MonitorNode) of
		pong -> 
			ok;
		pang ->
			io:format("CANNOT CONNECT TO MONITOR NODE: ~p!~n", [MonitorNode]),
			neuron_loop(MonitorNode, OutNeurons, InNeurons, Activation)
	end,
	receive
		{ input, Source, Value } -> 
			io:format("Received input ~w from ~p!~n", [Value, Source]);
		{ update, Source, Value } -> 
			io:format("Received update ~w from ~p!~n", [Value, Source])
	end,
	neuron_loop(MonitorNode, OutNeurons, InNeurons, Activation).



init_neuron_loop(MonitorNode, OutNeurons, Activation) -> 
	receive
		{out, InNeurons} -> 
			neuron_loop(MonitorNode, OutNeurons, InNeurons, Activation)
	end,
	init_neuron_loop(MonitorNode, OutNeurons, Activation).


init_layer(_, 0, NeuronPids, _, _) ->
	NeuronPids;
init_layer(Node, N, NeuronPids, Activation, NextNeurons) ->
	Pid = spawn(Node, deeplearn, init_neuron_loop, [Node, NextNeurons, Activation]),
	init_layer(Node, N-1, [Pid | NeuronPids], Activation, NextNeurons).

connect_out_neurons([_]) -> ok;
connect_out_neurons([Layer, InLayer | Tail]) ->
	lists:map(fun(N) -> N ! {out, Layer} end, InLayer),
	connect_out_neurons([InLayer | Tail]).


spawn_nn_helper(_, [], _, [LastLayer | TailNeurons]) -> 
	lists:map(fun(N) -> N ! {out, []} end, LastLayer),
	connect_out_neurons([LastLayer | TailNeurons]);	
spawn_nn_helper(Node, [H | TailLayers], Activation, []) ->
	NeuronPids = init_layer(Node, H, [], Activation, []),
	spawn_nn_helper(Node, TailLayers, Activation, [NeuronPids]);
spawn_nn_helper(Node, [H | TailLayers], Activation, [NextNeurons | PrevNeurons]) ->
	NeuronPids = init_layer(Node, H, [], Activation, NextNeurons),
	spawn_nn_helper(Node, TailLayers, Activation,
					[NeuronPids | [ NextNeurons | PrevNeurons ]]).


spawn_nn(Node, Layers, Activation) ->
	io:format("Spawning a neural network on ~p with layers ~w....~n", [Node, Layers]),
	spawn_nn_helper(Node, Layers, Activation, []).

