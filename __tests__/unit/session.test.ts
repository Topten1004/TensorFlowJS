import TFModel from "../../src/utils/tf_model";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { Sequential } from "@tensorflow/tfjs-node-gpu";

describe("Predict values", () => {
  
  it("Zero", async () => {
    const tfModel: TFModel | null = await TFModel.load("file://assets/models/trained_model/");
    if (tfModel == null) throw new Error("No model found");
    expect(Math.round(tfModel.predict(0) * 100) / 100).toBe(-1);
  });

  it("Two", async () => {
    const tfModel: TFModel | null = await TFModel.load("file://assets/models/trained_model/");
    if (tfModel == null) throw new Error("No model found");
    expect(Math.round(tfModel.predict(2) * 100) / 100).toBe(3);
  });

});