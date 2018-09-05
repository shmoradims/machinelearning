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

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform.Arguments), typeof(SignatureDataTransform),
    OnnxTransform.UserName, OnnxTransform.ShortName)]

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(OnnxTransform.OnnxMapper), null, typeof(SignatureLoadRowMapper),
    "", OnnxTransform.OnnxMapper.LoaderSignature)]

namespace Microsoft.ML.Transforms.Onnx
{
    public static class OnnxTransform
    {
        public sealed class Arguments : TransformInputBase
        {

            [Argument(ArgumentType.Required, HelpText = "Path to the onnx model file.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "TBD", SortOrder = 1)]
            public string InputColumn;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "TBD", SortOrder = 2)]
            public string OutputColumn;

            public OnnxModelInfo ModelInfo;
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

        internal sealed class OnnxMapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly Arguments _args;
            private OnnxModel _model;

            private readonly OnnxNodeInfo _outputNodeInfo;
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

            public OnnxMapper(IHostEnvironment env, ISchema inputSchema, Arguments transformArgs, OnnxModel model = null)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("OnnxMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _args = transformArgs;

                _model = model ?? new OnnxModel(transformArgs.ModelFile, transformArgs.ModelInfo);
                _idvToTensorAdapter = new IdvToTensorAdapter(inputSchema, transformArgs);
                _outputNodeInfo = transformArgs.ModelInfo.OutputsInfo[0];
                
                // TODO: Remove assumption below
                // Assume first output dimension is 1
                int[] dims = _outputNodeInfo.Shape.Skip(1).Select(x => (int)x).ToArray();
                var outputItemType = OnnxUtils.RawToMlNetType(_outputNodeInfo.DataType);
                _outputColType = new VectorType(outputItemType, dims);
                _outputColName = _outputNodeInfo.Name;
                _host.Assert(_outputNodeInfo.DataType == _outputColType.ItemType.RawType);
            }

            public static OnnxMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("OnnxModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();
                var model = new OnnxModel(modelBytes);

                var inputColumn = ctx.LoadNonEmptyString();
                var outputColumn = ctx.LoadNonEmptyString();
                var transformArg = new Arguments() { InputColumn = inputColumn, OutputColumn = outputColumn };

                return new OnnxMapper(env, schema, transformArg, model);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[1];
                info[0] = new RowMapperColumnInfo(_outputColName, _outputColType, null);
                return info;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && (_idvToTensorAdapter._idvColumnIndex == col);
            }

            public void Save(ModelSaveContext ctx)
            {
                _host.AssertValue(ctx);
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                ctx.SaveBinaryStream("OnnxModel", w => { w.WriteByteArray(_model.ToByteArray()); });
                ctx.SaveNonEmptyString(_args.InputColumn);
                ctx.SaveNonEmptyString(_args.OutputColumn);
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                var getters = new Delegate[1];
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = Utils.MarshalInvoke(MakeGetter<int>, _outputNodeInfo.DataType, input);

                    ch.Done();
                    return getters;
                }
            }

            private Delegate MakeGetter<T>(IRow input)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == _outputColType.ItemType.RawType);

                _idvToTensorAdapter.InitializeValueGetters(input);

                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var outputTensors = _model.Run(new List<Tensor> { _idvToTensorAdapter.GetTensor() });
                    Contracts.Assert(outputTensors.Count() > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    OnnxUtils.CopyTo(outputTensors[0], values);
                    dst = new VBuffer<T>(values.Length, values);
                };

                return valuegetter;
            }
        }

        /// <summary>
        /// IdvToTensorAdapter adapts an Idv (row-iterator interface) to a tensor-iterator interface.
        /// For an Idv, you'd need to create a cursor and iterate over it to get each rows of the Idv.
        /// After adaptation, you'd call GetTensor() on the IdvToTensorAdapter object to get the Tensor equivalent of 
        /// each row. 
        /// </summary>
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
                _onnxTensorShape = transformArgs.ModelInfo.InputsInfo[0].Shape;

                // TODO: Check that the idv and tensor sizes match
                // TODO: Check type matches
            }

            public void InitializeValueGetters(IRow idvRow)
            {
                var type = _idvColumnType.ItemType.RawType;
                _tensorValueGetter = Utils.MarshalInvoke(
                    CreateTensorValueGetter<int>, type, idvRow, _idvIsVectorColumn, _idvColumnIndex, _onnxTensorShape);
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

            private interface ITensorValueGetter
            {
                Tensor GetTensor();
            }

            private class TensorValueGetter<T> : ITensorValueGetter
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

            private class TensorValueGetterVec<T> : ITensorValueGetter
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
        }
    }
}
