import * as tf from "@tensorflow/tfjs-node-gpu";
import { ModelCompileArgs, Sequential, string } from "@tensorflow/tfjs-node-gpu";

class TFModel {
  private model: Sequential;
  private path: string;

  static async new(path: string, createModel: (model: Sequential) => Sequential): Promise<TFModel> {
    const loadedModel = await TFModel.load(path);
    if (loadedModel != null) return loadedModel;
    return new TFModel(path, createModel(tf.sequential()));
  }

  static async load(path: string) : Promise<TFModel | null> {
    try {
      return new TFModel(path, (await tf.loadLayersModel(`${path}/model.json`)) as Sequential);
    } catch {
      return null;
    }
  }

  constructor(path: string, model: Sequential) {
    this.path = path;
    this.model = model;
  }

  getModel(): Sequential {
    return this.model;
  }

  predict(value: number) {
    const result = this.model.predict(tf.tensor1d([value])) as tf.Tensor;
    return result.dataSync()[0];
  }

  save() {
    this.model.save(this.path);
  }
}

export default TFModel;