using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Scoring;
using OnnxShape = System.Collections.Generic.List<long>;

namespace Microsoft.ML.Transforms.Onnx
{
    public class OnnxUtils
    {
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
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }

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
                throw new NotImplementedException($"Unsupported type {typeof(T)}");
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
