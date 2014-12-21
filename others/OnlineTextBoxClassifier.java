import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Random;

public class OnlineTextBoxClassifier {

  private static final int NUM_LABELS = 33;
  private static final int NUM_FEATURES = (int) Math.pow(2, 24);
  private static final int RAND_SEED = 42;
  private static final double ALPHA = 0.1;

  private String trainLoc;
  private String trainLabelsLoc;
  private double[] modelWeights;
  private double[] gradientSum;
  private int folds;
  private long time;

  public OnlineTextBoxClassifier(int f, String tl, String tll) {
    folds = f;
    trainLoc = tl;
    trainLabelsLoc = tll;
    time = System.currentTimeMillis();
  }

  private void crossValidate() throws IOException {
    for (int f = 0; f < folds; ++f) {
      modelWeights = new double[NUM_LABELS * NUM_FEATURES]; // we want new models for every fold
      gradientSum = new double[NUM_LABELS * NUM_FEATURES];
      double loss = iterateOnDataset(true, f);
      System.err.println("Training on fold " + f + " the log-loss was " + loss);

      if (folds > 1) { // doing the separate testing step only makes sense if we had more than one folds
        loss = iterateOnDataset(false, f);
        System.err.println("Testing on fold " + f + " the log-loss was " + loss);
      }
    }
  }

  /**
   * @param train
   *          Indicates whether it is a train iteration, i.e. whether the update of the weights should happen.
   * @param fold
   * @return
   * @throws IOException
   */
  private double iterateOnDataset(boolean train, int fold) throws IOException {
    Random r = new Random(RAND_SEED); // every time we go over the file, we wish to see the same 'randomness'
    BufferedReader trainingDataReader = new BufferedReader(new InputStreamReader(new FileInputStream(trainLoc)));
    BufferedReader trainingLabelsReader = new BufferedReader(new InputStreamReader(new FileInputStream(trainLabelsLoc)));
    trainingDataReader.readLine(); // the first lines are not interesting at all
    trainingLabelsReader.readLine();

    String line;
    double loss = 0;
    int lineCounter = 0;
    while ((line = trainingDataReader.readLine()) != null) {
      double[] groundTruth = line2Vec(trainingLabelsReader.readLine(), true);
      int randFold = r.nextInt(folds);
      if (folds == 1 || (train && randFold != fold) || (!train && randFold == fold)) {
        double[] features = line2Vec(line, false);
        double[] predictions = predict(features);
        loss += calculateLossOnInstance(predictions, groundTruth);

        updateWeights(features, predictions, groundTruth, train);

        if (++lineCounter % 50_000 == 0) {
          double elapsed = (System.currentTimeMillis() - time) / 1000.0d;
          System.err.println("Avg. log loss over " + lineCounter + " instances: " + (loss / NUM_LABELS / lineCounter)
              + " in " + elapsed + " secs.");
        }
      }
    }
    trainingDataReader.close();
    trainingLabelsReader.close();
    return loss / (NUM_LABELS * lineCounter);
  }

  private void updateWeights(double[] f, double[] pred, double[] y, boolean train) {
    if (train) {
      for (int c = 0; c < NUM_LABELS; ++c) {
        if (c != 13) {// this is because we do not wish to model class 14
          for (int i = 0; i < f.length; ++i) {
            double diff = pred[c] - y[c];
            gradientSum[c * NUM_FEATURES + (int) f[i]] += Math.abs(diff);
            modelWeights[c * NUM_FEATURES + (int) f[i]] -= diff * ALPHA
                / Math.sqrt(gradientSum[c * NUM_FEATURES + (int) f[i]]);
          }
        }
      }
    }
  }

  private double[] predict(double[] features) {
    double[] predictions = new double[NUM_LABELS];
    for (int c = 0; c < NUM_LABELS; ++c) {
      if (c != 13) { // this is because we do not wish to model class 14
        double dotProduct = 0.0d;
        for (int i = 0; i < features.length; ++i) {
          dotProduct += modelWeights[c * NUM_FEATURES + (int) features[i]];
        }
        predictions[c] = 1.0 / (1 + Math.exp(-Math.max(Math.min(dotProduct, 20), -20)));
      }
    }
    return predictions;
  }

  private double calculateLossOnInstance(double[] predictions, double[] etalon) {
    double loss = 0.0d;
    for (int c = 0; c < etalon.length; ++c) {
      double p = Math.max(Math.min(predictions[c], 1 - 1E-15), 1E-15);
      loss -= Math.log(etalon[c] == 1 ? p : (1 - p));
    }
    return loss;
  }

  private double[] line2Vec(String line, boolean lineWithLabels) {
    String[] parts = line.split(",");
    double[] vals = new double[parts.length - 1];
    for (int i = 1; i < parts.length; ++i) {
      vals[i - 1] = lineWithLabels ? Double.parseDouble(parts[i]) : Math.abs((i + parts[i]).hashCode()) % NUM_FEATURES;
    }
    return vals;
  }

  private void writeTestPredictions(String testFile) throws IOException {
    String line;
    BufferedReader testDataReader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile)));
    testDataReader.readLine();
    PrintWriter out = new PrintWriter("prediction.csv");
    out.println("id_label,pred");
    while ((line = testDataReader.readLine()) != null) {
      int id = Integer.parseInt(line.split(",")[0]);
      double[] predictions = predict(line2Vec(line, false));
      for (int c = 0; c < predictions.length; ++c) {
        out.printf("%d_y%d,%.9f%n", id, (c + 1), predictions[c]);
      }
    }
    testDataReader.close();
    out.close();
  }

  public static void main(String[] args) {
    if (args.length < 2) {
      System.err.println("The program must recieve at least 2 command line arguments, i.e.");
      System.err.println(" - the location of the training data\n - the location of the training labels");
      System.err.println(" - the location of the test data (optional)");
      System.exit(3);
    }
    String trainLocation = args[0], trainLabelsLocation = args[1];
    int numOfFolds = 1;
    OnlineTextBoxClassifier otbc = new OnlineTextBoxClassifier(numOfFolds, trainLocation, trainLabelsLocation);
    try {
      otbc.crossValidate();
      if (args.length > 2) {
        otbc.writeTestPredictions(args[2]);
      } else {
        System.err
            .println("A 3rd command line argument (with the location of the test data) is expected to predict the test labels.");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}