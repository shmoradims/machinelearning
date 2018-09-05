using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Scoring;
using OnnxShape = System.Collections.Generic.List<long>;

namespace Microsoft.ML.Transforms.Onnx
{
    /// <summary>
    /// OnnxModelInfo contains the data that we should get from 
    /// Sonoma API once that functionality is added.
    /// </summary>
    public sealed class OnnxModelInfo
    {
        public OnnxNodeInfo[] InputsInfo;
        public OnnxNodeInfo[] OutputsInfo;
    }

    /// <summary>
    /// OnnxNodeInfo contains all the information for a given node (e.g. inputs/outputs)
    /// of an Onnx model.
    /// </summary>
    public class OnnxNodeInfo
    {
        public string Name;
        public OnnxShape Shape;
        public Type DataType = typeof(System.Single);
    }

    /// <summary>
    /// OnnxModel is a facad for ModelManager. ModelManager is provided by Sonoma API, 
    /// and it has a lot of functionality (multiple models, multiple versions) that are not
    /// needed by Onnx transform, which only needs a single model. This facad simplifies the
    /// usage of onnx model. 
    /// </summary>
    internal sealed class OnnxModel
    {
        private static readonly int IgnoredVersion = int.MaxValue;
        private ModelManager _modelManager;
        private string _modelName;
        private readonly List<string> _inputNames;
        private readonly List<string> _outputNames;

        public OnnxModel(string modelFile, OnnxModelInfo modelInfo)
        {
            // Load the onnx model
            var modelFileInfo = new FileInfo(modelFile);
            _modelName = Path.GetFileNameWithoutExtension(modelFileInfo.Name);
            _modelManager = new ModelManager(modelFileInfo.Directory.FullName, true);
            _modelManager.InitModel(_modelName, IgnoredVersion);

            _inputNames = modelInfo.InputsInfo.Select(i => i.Name).ToList();
            _outputNames = modelInfo.OutputsInfo.Select(i => i.Name).ToList();
        }

        public OnnxModel(byte[] modelBytes)
        {
            throw new NotImplementedException("Need an API to serialize/deserialize onnx models to byte arrays!");
        }

        public List<Tensor> Run(List<Tensor> inputTensors)
        {
            var outputTensors = _modelManager.RunModel(
                _modelName, IgnoredVersion, _inputNames, inputTensors, _outputNames);

            return outputTensors;
        }

        public byte[] ToByteArray()
        {
            throw new NotImplementedException("Need an API to serialize/deserialize onnx models to byte arrays!");
        }
    }

    internal sealed class OnnxUtils
    {
        /// <summary>
        /// Sonoma API only provides Tensor() constructors with overloaded versions
        /// based on data type. ML.NET cannot use the overloaded version and requires 
        /// generic version. CreateScalarTensor<T> is generic wrapper on top of
        /// overloaded Tensor(T data) constructors.
        /// </summary>
        public static Tensor CreateScalarTensor<T>(T data)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new Tensor((System.Boolean)(object)data);
            }
            else if (typeof(T) == typeof(System.Byte))
            {
                return new Tensor((System.Byte)(object)data);
            }
            else if (typeof(T) == typeof(System.Char))
            {
                return new Tensor((System.Char)(object)data);
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new Tensor((System.Double)(object)data);
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new Tensor((System.Single)(object)data);
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new Tensor((System.Int32)(object)data);
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new Tensor((System.Int64)(object)data);
            }
            else if (typeof(T) == typeof(System.SByte))
            {
                return new Tensor((System.SByte)(object)data);
            }
            else if (typeof(T) == typeof(System.Int16))
            {
                return new Tensor((System.Int16)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt32))
            {
                return new Tensor((System.UInt32)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt64))
            {
                return new Tensor((System.UInt64)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt16))
            {
                return new Tensor((System.UInt16)(object)data);
            }
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }

        /// <summary>
        /// Sonoma API only provides Tensor() constructors with overloaded versions
        /// based on data type. ML.NET cannot use the overloaded version and requires 
        /// generic version. CreateTensor<T> is generic wrapper on top of
        /// overloaded Tensor(T[] data, OnnxShape shape) constructors.
        /// </summary>
        public static Tensor CreateTensor<T>(T[] data, OnnxShape shape)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new Tensor(((System.Boolean[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new Tensor(((System.Double[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new Tensor(((System.Single[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new Tensor(((System.Int32[])(object)data).ToList(), shape);
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new Tensor(((System.Int64[])(object)data).ToList(), shape);
            }
            throw new NotImplementedException($"Not implemented type {typeof(T)}");
        }

        /// <summary>
        /// Sonoma API only provides CopyTo() functions with overloaded versions
        /// based on data type. ML.NET cannot use the overloaded version and requires 
        /// generic version. CopyTo<T> is generic wrapper on top of
        /// overloaded Tensor.CopyTo(List<T> dst) methods.
        /// Also Tensor.CopyTo(List<T> dst) requires a list input, whereas ML.NET
        /// provides array buffers to copy values to. This mismatch causes an extra copy.
        /// </summary>
        public static void CopyTo<T>(Tensor tensor, T[] dst)
        {
            if (typeof(T) == typeof(System.Single))
            {
                // Sonoma only takes List<T>. We need to do an extra copy to T[]
                var listDst = new List<System.Single>();
                var typedDst = (System.Single[])(object)dst;
                tensor.CopyTo(listDst);
                listDst.CopyTo(typedDst);
            }
            else
                throw new NotImplementedException($"Not implemented type {typeof(T)}");
        }

        public static PrimitiveType RawToMlNetType(Type type)
        {
            // Todo: is there already a utility to do this?
            DataKind kind;
            if (type == typeof(System.Single))
                kind = DataKind.R4;
            else if (type == typeof(System.Double))
                kind = DataKind.R8;
            else if (type == typeof(System.UInt32))
                kind = DataKind.U4;
            else if (type == typeof(System.UInt64))
                kind = DataKind.U8;
            else
                throw new NotSupportedException("Type not supported.");

            return PrimitiveType.FromKind(kind);
        }
    }
}
