// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Scoring;
using Microsoft.ML.Transforms.Onnx;
using OnnxShape = System.Collections.Generic.List<long>;
using IgnoreType = System.Int32;

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform.Arguments), typeof(SignatureDataTransform),
    OnnxTransform.UserName, OnnxTransform.ShortName)]

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(OnnxTransform.OnnxMapper), null, typeof(SignatureLoadRowMapper),
    "", OnnxTransform.OnnxMapper.LoaderSignature)]

namespace Microsoft.ML.Transforms.Onnx
{
    // OnnxModelMetadata contains the data that we should get from 
    // Sonoma API once that functionality is added.
    // For V0, we're working with models that have single input and single output.
    public class OnnxModelMetadata
    {
        public string InputNodeName;
        public string OutputNodeName;
        public OnnxShape InputNodeShape;
        public OnnxShape OutputNodeShape;
        public Type InputNodeType = typeof(System.Single);
        public Type OutputNodeType = typeof(System.Single);
    }

    public static class OnnxTransform
    {
        internal interface ITensorValueGetter
        {
            Tensor GetTensor();
        }

        internal class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;

            public TensorValueGetter(IRow input, int colIndex)
            {
                _srcgetter = input.GetGetter<T>(colIndex);
            }
            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                return OnnxUtils.CreateScalarTensor(scalar);
            }
        }

        internal class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly OnnxShape _tensorShape;
            private VBuffer<T> _vBuffer;
            private VBuffer<T> _vBufferDense;
            public TensorValueGetterVec(IRow input, int colIndex, OnnxShape tensorShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                _tensorShape = tensorShape;
                _vBuffer = default;
                _vBufferDense = default;
            }
            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyToDense(ref _vBufferDense);
                return OnnxUtils.CreateTensor(_vBufferDense.Values, _tensorShape);
            }
        }

        internal sealed class IdvToTensorAdapter
        {
            // Idv information
            private readonly string _idvColumnName;
            internal readonly int _idvColumnIndex;
            private readonly bool _idvIsVectorColumn;
            public readonly ColumnType _idvColumnType;

            // Onnx tensor information
            private readonly OnnxShape _onnxTensorShape;

            private ITensorValueGetter _tensorValueGetter;

            public IdvToTensorAdapter(ISchema idvSchema, Arguments transformArgs)
            {
                _idvColumnName = transformArgs.InputColumn;
                if (!idvSchema.TryGetColumnIndex(_idvColumnName, out _idvColumnIndex))
                    throw Contracts.Except($"Column '{_idvColumnName}' does not exist");
                _idvColumnType = idvSchema.GetColumnType(_idvColumnIndex);
                _idvIsVectorColumn = _idvColumnType.IsVector;
                _onnxTensorShape = transformArgs.ModelMetadata.InputNodeShape;

                // TODO: Check that the idv and tensor sizes match
                // TODO: Check type matches
            }

            public void InitializeValueGetters(IRow idvRow)
            {
                var type = _idvColumnType.ItemType.RawType;
                _tensorValueGetter = Utils.MarshalInvoke(
                    CreateTensorValueGetter<IgnoreType>, type, idvRow, _idvIsVectorColumn, _idvColumnIndex, _onnxTensorShape);
            }

            public Tensor GetTensor()
            {
                return _tensorValueGetter.GetTensor();
            }

            private ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, OnnxShape tensorShape)
            {
                if (isVector)
                    return new TensorValueGetterVec<T>(input, colIndex, tensorShape);
                else
                    return new TensorValueGetter<T>(input, colIndex);
            }

            //public void InitializeValueGetters<T>(IRow idvRow)
            //{
            //    if (_idvIsVectorColumn)
            //        _tensorValueGetter = new TensorValueGetterVec<T>(idvRow, _idvColumnIndex, _onnxTensorShape);
            //    else
            //        _tensorValueGetter = new TensorValueGetter<T>(idvRow, _idvColumnIndex);
            //}
        }
        internal sealed class OnnxMapper : IRowMapper
        {
            private readonly IHost _host;
            private ModelManager _modelManager;
            private OnnxModelMetadata _modelMetadata;
            private string _modelName;

            private static readonly int IgnoredVersion = int.MaxValue;

            private readonly List<string> _inputNames;
            private readonly List<string> _outputNames;
            private readonly List<long> _inputShape;
            private readonly List<long> _outputShape;

            private readonly ColumnType _outputColType;
            private readonly string _outputColName;

            private readonly IdvToTensorAdapter _idvToTensorAdapter;

            public const string LoaderSignature = "OnnxMapper";
            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "ONNX",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }

            public OnnxMapper(IHostEnvironment env, ISchema inputSchema, Arguments transformArgs)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("OnnxMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));

                // Load the onnx model
                var modelFileInfo = new FileInfo(transformArgs.ModelFile);
                _modelName = Path.GetFileNameWithoutExtension(modelFileInfo.Name);
                _modelManager = new ModelManager(modelFileInfo.Directory.FullName, true);
                _modelManager.InitModel(_modelName, IgnoredVersion);

                _idvToTensorAdapter = new IdvToTensorAdapter(inputSchema, transformArgs);

                _modelMetadata = transformArgs.ModelMetadata;
                _inputNames = new List<string> { _modelMetadata.InputNodeName };
                _outputNames = new List<string> { _modelMetadata.OutputNodeName };
                _inputShape = _modelMetadata.InputNodeShape;
                _outputShape = _modelMetadata.OutputNodeShape;

                // TODO: Remove assumption below
                // Assume first output dimension is 1
                int[] dims = _outputShape.Skip(1).Select(x => (int)x).ToArray();
                var outputItemType = OnnxUtils.RawToMlNetType(_modelMetadata.OutputNodeType);
                _outputColType = new VectorType(outputItemType, dims);
                _outputColName = _outputNames[0];
            }

            public static OnnxMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                var numInputs = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(numInputs > 0);

                string[] source = new string[numInputs];
                for (int j = 0; j < source.Length; j++)
                    source[j] = ctx.LoadNonEmptyString();

                byte[] data = null;
                if (!ctx.TryLoadBinaryStream("OnnxModel", r => data = r.ReadByteArray()))
                    throw env.ExceptDecode();

                var outputColName = ctx.LoadNonEmptyString();

                return new OnnxMapper(env, schema, data, source, outputColName);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[1];
                info[0] = new RowMapperColumnInfo(_outputColName, _outputColType, null);
                return info;
            }
            public void Save(ModelSaveContext ctx)
            {
                //_host.AssertValue(ctx);
                //ctx.CheckAtModel();
                //ctx.SetVersionInfo(GetVersionInfo());

                //var buffer = new TFBuffer();
                //_session.Graph.ToGraphDef(buffer);

                //ctx.SaveBinaryStream("TFModel", w =>
                //{
                //    w.WriteByteArray(buffer.ToArray());
                //});
                //Contracts.AssertNonEmpty(InputColNames);
                //ctx.Writer.Write(InputColNames.Length);
                //foreach (var colName in InputColNames)
                //    ctx.SaveNonEmptyString(colName);

                //ctx.SaveNonEmptyString(OutputColName);
            }

            //            private TFSession LoadTFSession(byte[] modelBytes, string modelArg)
            //            {
            //                var graph = new TFGraph();
            //                try
            //                {
            //                    graph.Import(modelBytes, "");
            //                }
            //                catch (Exception ex)
            //                {
            //                    if (!string.IsNullOrEmpty(modelArg))
            //                        throw _host.Except($"Onnx exception triggered while loading model from '{modelArg}'");
            //#pragma warning disable MSML_NoMessagesForLoadContext
            //                    throw _host.ExceptDecode(ex, "Onnx exception triggered while loading model.");
            //#pragma warning restore MSML_NoMessagesForLoadContext

            //                }
            //                return new TFSession(graph);
            //            }

            //            private ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, TFShape tfShape)
            //            {
            //                if (isVector)
            //                    return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            //                else
            //                    return new TensorValueGetter<T>(input, colIndex);
            //            }

            private Delegate MakeGetter(IRow input)
            {
                _host.Assert(_modelMetadata.OutputNodeType == _outputColType.ItemType.RawType);
                return Utils.MarshalInvoke(MakeGetter<IgnoreType>, _modelMetadata.OutputNodeType, input);
            }

            private Delegate MakeGetter<T>(IRow input)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == _outputColType.ItemType.RawType);

                _idvToTensorAdapter.InitializeValueGetters(input);

                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    List<Tensor> outputTensors = _modelManager.RunModel(
                        _modelName,
                        IgnoredVersion,
                        _inputNames,
                        new List<Tensor> { _idvToTensorAdapter.GetTensor() },
                        _outputNames
                    );

                    Contracts.Assert(outputTensors.Count() > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    OnnxUtils.CopyTo(outputTensors[0], values);
                    dst = new VBuffer<T>(values.Length, values);
                };

                return valuegetter;
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                var getters = new Delegate[1];
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = MakeGetter(input);
                    ch.Done();
                    return getters;
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && (_idvToTensorAdapter._idvColumnIndex == col);
            }

            //private static (ColumnType, TFDataType) GetOutputTypes(TFGraph graph, string columnName)
            //{
            //    var tfoutput = new TFOutput(graph[columnName]);
            //    var shape = graph.GetTensorShape(tfoutput);

            //    int[] dims = new int[shape.NumDimensions];
            //    for (int k = 0; k < shape.NumDimensions; k++)
            //        dims[k] = (int)(shape[k] == -1 ? 1 : shape[k]);

            //    var type = OnnxUtils.Tf2MlNetType(tfoutput.OutputType);
            //    return (new VectorType(type, dims), tfoutput.OutputType);
            //}

            //private static (string[], int[], bool[], TFShape[], TFDataType[]) GetInputMetaData(TFGraph graph, string[] source, ISchema inputSchema)
            //{
            //    var tfShapes = new TFShape[source.Length];
            //    var tfTypes = new TFDataType[source.Length];
            //    var colNames = new string[source.Length];
            //    var inputColIndices = new int[source.Length];
            //    var isInputVector = new bool[source.Length];
            //    for (int j = 0; j < source.Length; j++)
            //    {
            //        colNames[j] = source[j];
            //        if (!inputSchema.TryGetColumnIndex(colNames[j], out inputColIndices[j]))
            //            throw Contracts.Except($"Column '{colNames[j]}' does not exist");

            //        isInputVector[j] = inputSchema.GetColumnType(inputColIndices[j]).IsVector;

            //        var tfoutput = new TFOutput(graph[colNames[j]]);

            //        if (!OnnxUtils.IsTypeSupported(tfoutput.OutputType))
            //            throw Contracts.Except($"Input type '{tfoutput.OutputType}' of input column '{colNames[j]}' is not supported in Onnx");

            //        tfShapes[j] = graph.GetTensorShape(tfoutput);
            //        tfTypes[j] = tfoutput.OutputType;

            //        var l = new long[tfShapes[j].NumDimensions];
            //        for (int ishape = 0; ishape < tfShapes[j].NumDimensions; ishape++)
            //        {
            //            l[ishape] = tfShapes[j][ishape] == -1 ? 1 : tfShapes[j][ishape];
            //        }
            //        tfShapes[j] = new TFShape(l);
            //    }
            //    return (colNames, inputColIndices, isInputVector, tfShapes, tfTypes);
            //}
        }

        public sealed class Arguments : TransformInputBase
        {

            [Argument(ArgumentType.Required, HelpText = "Path to the onnx model file.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "TBD", SortOrder = 1)]
            public string InputColumn;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "TBD", SortOrder = 2)]
            public string OutputColumn;

            public OnnxModelMetadata ModelMetadata;
        }

        public const string Summary = "Transforms the data using the Onnx model.";
        public const string UserName = "OnnxTransform";
        public const string ShortName = "Onnx";
        private const string RegistrationName = "OnnxTransform";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen Onnx model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="name">Name of the output column. Keep it same as in the Onnx model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Onnx model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string inputColumn, string outputColumn)
        {
            return Create(env, new Arguments() { ModelFile = modelFile, InputColumn = inputColumn, OutputColumn = outputColumn }, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckNonWhiteSpace(args.InputColumn, nameof(args.InputColumn));
            host.CheckNonWhiteSpace(args.OutputColumn, nameof(args.OutputColumn));
            host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
            host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));

            var mapper = new OnnxMapper(host, input.Schema, args);
            return new RowToRowMapperTransform(host, input, mapper);
        }
    }
}
