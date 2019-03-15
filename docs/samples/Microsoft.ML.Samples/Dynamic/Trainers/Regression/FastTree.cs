using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Regression
{
    public static class FastTree
    {
        // This example requires installation of Microsoft.ML.FastTree NuGet package:
        // https://www.nuget.org/packages/Microsoft.ML.FastTree/
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training examples.
            var examples = GenerateRandomDataPoints(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(examples);

            // Define the trainer.
            var pipeline = mlContext.Regression.Trainers.FastTree();

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing examples. Use different random seed to make it different from training data.
            var testingData = mlContext.Data.LoadFromEnumerable(GenerateRandomDataPoints(500, seed:123));

            var predictions = model.Transform(testingData);

            // Look at 10 predictions
            var trueLabels = predictions.GetColumn<float>("Label").Take(10).ToArray();
            var predictedLabels = predictions.GetColumn<float>("Score").Take(10).ToArray();
            for (int i = 0; i < 10; i++)
                Console.WriteLine($"Label: {trueLabels[i]:F3}, Prediction: {predictedLabels[i]:F3}");

            // Expected output:
            //   Label: 0.985, Prediction: 0.938
            //   Label: 0.155, Prediction: 0.131
            //   Label: 0.515, Prediction: 0.517
            //   Label: 0.566, Prediction: 0.519
            //   Label: 0.096, Prediction: 0.089
            //   Label: 0.061, Prediction: 0.050
            //   Label: 0.078, Prediction: 0.073
            //   Label: 0.509, Prediction: 0.438
            //   Label: 0.549, Prediction: 0.558
            //   Label: 0.721, Prediction: 0.494

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(predictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   Mean Absolute Error: 0.05
            //   Mean Squared Error: 0.00
            //   Root Mean Squared Error: 0.06
            //   RSquared: 0.95
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed=0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with label.
                    Features = Enumerable.Repeat(label, 50).Select(x => x + randomFloat()).ToArray()
                };
            }
        }

        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }
    }
}