import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeuralNetwork {

    Matrix weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput;
    double l_rate = 0.01;
    Map<Integer, Integer> counterMap = new HashMap<>();

    public NeuralNetwork(int input, int hidden, int output) {
        weightsInputHidden = new Matrix(hidden, input);
        weightsHiddenOutput = new Matrix(output, hidden);

        biasHidden = new Matrix(hidden, 1);
        biasOutput = new Matrix(output, 1);

    }

    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weightsInputHidden, input);
        hidden.add(biasHidden);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weightsHiddenOutput, hidden);
        output.add(biasOutput);
        output.sigmoid();

        return output.toArray();
    }


    public void fit(double[][] X, double[][] Y, int epochs) {
        for (int i = 0; i < epochs; i++) {
            int sampleN = (int) (Math.random() * X.length);
            int oldValue = counterMap.getOrDefault(sampleN, 0);
            counterMap.put(sampleN, ++oldValue);
            this.train(X[sampleN], Y[sampleN]);
        }
        System.out.println("Length: "+X.length);
    }

    public void train(double[] X, double[] Y) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weightsInputHidden, input);
        hidden.add(biasHidden);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weightsHiddenOutput, hidden);
        output.add(biasOutput);
        output.sigmoid();

        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.subtract(target, output);
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(l_rate);

        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta = Matrix.multiply(gradient, hidden_T);

        weightsHiddenOutput.add(who_delta);
        biasOutput.add(gradient);

        Matrix who_T = Matrix.transpose(weightsHiddenOutput);
        Matrix hidden_errors = Matrix.multiply(who_T, error);

        Matrix h_gradient = hidden.dsigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(l_rate);

        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

        weightsInputHidden.add(wih_delta);
        biasHidden.add(h_gradient);

    }

    void printCounterMap() {
        System.out.println("Distribution of training hits");
        for (Map.Entry<Integer, Integer> entry : counterMap.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    void printWeights(){
        System.out.println("WeightsInputHidden: ");
        weightsInputHidden.print();
        System.out.println("BiasHidden: ");
        biasHidden.print();
        System.out.println("BiasOutput: ");
        biasOutput.print();
        System.out.println("WeightsHiddenOutput: ");
        weightsHiddenOutput.print();
    }

}