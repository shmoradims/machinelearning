using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using OnnxShape = System.Collections.Generic.List<long>;


namespace Microsoft.ML.Transforms.Onnx
{
    class Program
    {
        private class TestData
        {
            [VectorType(784)]
            public float[] pixels;
        }
        static void Main(string[] args)
        {
            TestTransform();
        }

        static void TestTransform()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var data = new[] { new TestData() { pixels = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 112, 254, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 96, 3, 0, 0, 0, 44, 238, 253, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 254, 76, 0, 0, 0, 224, 253, 168, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 204, 229, 33, 0, 22, 152, 254, 212, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 98, 0, 0, 62, 253, 254, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 254, 231, 0, 0, 71, 246, 231, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 241, 250, 68, 0, 10, 181, 253, 147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 159, 253, 198, 23, 0, 147, 253, 195, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 253, 240, 99, 0, 85, 250, 240, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 254, 253, 205, 113, 195, 254, 253, 167, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 209, 255, 254, 254, 254, 254, 255, 254, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 145, 253, 254, 253, 253, 253, 253, 249, 121, 247, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 169, 253, 253, 254, 179, 78, 240, 253, 165, 0, 54, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 253, 253, 237, 57, 3, 88, 251, 237, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 253, 147, 36, 0, 8, 205, 253, 118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 239, 254, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 112, 253, 209, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 254, 239, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 248, 222, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 253, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } };
                var dataIdv = ComponentCreation.CreateDataView(env, data);

                
                var modelFile = @"F:\ProjectsData\Sonoma\PoC\mlnet_dummy\OnnxTransformTempSolution\OnnxModel\lotusrt_cntk_fullyconnect.onnx";
                var modelMetadata = new OnnxModelInfo()
                {
                    InputsInfo = new OnnxNodeInfo[]
                    {
                        new OnnxNodeInfo
                        {
                            Name = "input0",
                            Shape = new OnnxShape { 1, 784 },
                        }
                    },

                    OutputsInfo = new OnnxNodeInfo[]
                    {
                        new OnnxNodeInfo
                        {
                            Name = "output0",
                            Shape = new OnnxShape { 1, 10 },
                        }
                    },
                };

                var transformArgs = new OnnxTransform.Arguments() { ModelFile = modelFile, InputColumn = "pixels", OutputColumn = "pixelsOut", ModelInfo = modelMetadata };

                var transform = OnnxTransform.Create(env, transformArgs, dataIdv);
                using (var cursor = transform.GetRowCursor(a => true))
                {
                    cursor.MoveNext();
                    var cgetter = cursor.GetGetter<VBuffer<float>>(1);
                    VBuffer<float> outTensor = default;
                    cgetter(ref outTensor);

                    var expectedOuput = new[] { 3.68102e-07, 3.22787e-05, 1.06563e-06, 1.24394e-09, 0.99871, 3.23013e-05, 0.0012081, 1.19806e-05, 3.39954e-06, 1.00852e-06 };
                    Debug.Assert(outTensor.Values.Length == expectedOuput.Length);
                    for (int i = 0; i < expectedOuput.Length; i++)
                    {
                        Debug.Assert(Math.Abs(outTensor.Values[i] - expectedOuput[i]) < 1e-6);
                    }
                }
            }
        }
    }
}
