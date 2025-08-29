
inputs = {{0,0},{0,1},{1,0},{1,1}};


labelsAND = {0,0,0,1};
labelsOR  = {0,1,1,1};
labelsXOR = {0,1,1,0};


netAND = NetChain[{LinearLayer[2], SoftmaxLayer[]},
   "Input"->2, "Output"->NetDecoder[{"Class",{0,1}}]];
andTr = NetTrain[netAND, Thread[inputs->labelsAND], "TrainedNet",
   ValidationSet->None, MaxTrainingRounds->300];


netOR = NetChain[{LinearLayer[2], SoftmaxLayer[]},
   "Input"->2, "Output"->NetDecoder[{"Class",{0,1}}]];
orTr = NetTrain[netOR, Thread[inputs->labelsOR], "TrainedNet",
   ValidationSet->None, MaxTrainingRounds->300];


netXOR = NetChain[{LinearLayer[4], Tanh, LinearLayer[2], SoftmaxLayer[]},
   "Input"->2, "Output"->NetDecoder[{"Class",{0,1}}]];
xorTr = NetTrain[netXOR, Thread[inputs->labelsXOR], "TrainedNet",
   ValidationSet->None, MaxTrainingRounds->600];


Print["AND: ", andTr /@ inputs];
Print["OR:  ", orTr /@ inputs];
Print["XOR: ", xorTr /@ inputs];
