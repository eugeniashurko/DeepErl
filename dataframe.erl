% Small module for 2d arrays based on 
% https://stackoverflow.com/questions/536753/using-twomulti-dimensional-array-in-erlang
-module(dataframe).
-export([new/2, get/3, set/4]).

new(Rows, Cols)->
    A = array:new(Rows),
    array:map(fun(_X, _T) -> array:new(Cols) end, A).

get(column, ColI, A) ->
	array:map(
		fun(Index, X) -> array:get(ColI, X) end,
		A);
get(row, RowI, A) ->
	array:get(RowI, A);
get(RowI, ColI, A) ->
	Row = array:get(RowI, A),
    array:get(ColI, Row).

set(RowI, ColI, Ele, A) ->
    Row = array:get(RowI, A),
    Row2 = array:set(ColI, Ele, Row),
    array:set(RowI, Row2, A).
