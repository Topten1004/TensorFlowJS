import * as tf from "@tensorflow/tfjs-node-gpu";
import { Sequential } from "@tensorflow/tfjs-node-gpu";
import TFModel from "./utils/tf_model";
import { plot, Plot } from "nodeplotlib";

async function main() {
  const tfModel: TFModel = await TFModel.new("file://assets/models/trained_model/", (model: Sequential) => {
    model.add(tf.layers.dense({ units: 10, inputShape: [1] }));
    model.add(tf.layers.dense({ units: 1 }));
    return model;
  });

  tfModel.getModel()
  
  tfModel.getModel().compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
    metrics: ["MAE"],
  });

  const range = (start: number, stop: number): number[] => Array.from({ length: stop - start + 1 }, (_, i) => start + i);
  const xTrain = range(-4, 4);
  const yTrain = xTrain.map(e => 2 * e - 1);
  
  const data: Plot[] = [
    {
      x: xTrain,
      y: yTrain,
      type: 'scatter',
    },
  ];
  
  plot(data);

  await tfModel.getModel().fit(
    tf.tensor1d(xTrain),
    tf.tensor1d(yTrain), {
    epochs: 1000,
    verbose: 1
  });

  tfModel.save();
}

main();
