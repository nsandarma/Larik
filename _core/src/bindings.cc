#include "../include/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
void bind_tensor(py::module &m, const std::string &type_name) {
  using TensorT = Tensor<T>;
  std::string py_class_name = "Tensor" + type_name;

  auto cls =
      py::class_<TensorT>(m, py_class_name.c_str())
          .def(py::init<const std::vector<size_t> &, std::vector<T> &&>(),
               py::arg("shape"), py::arg("buffer"))
          .def("size", &TensorT::size)
          .def("shape", &TensorT::shape)
          .def("dim", &TensorT::dim)

          .def("__getitem__",
               [](const TensorT &self, size_t i) { return self[i]; })
          .def("__setitem__",
               [](TensorT &self, size_t i, T value) { self[i] = value; })

          .def("__call__",
               [](TensorT &self, const std::vector<size_t> &indices) {
                 return self(indices);
               })
          .def("__call__",
               [](const TensorT &self, const std::vector<size_t> &indices) {
                 return self(indices);
               })

          // ✅ versi 2D
          .def_static(
              "from_nested",
              [](const std::vector<std::vector<T>> &nested) {
                std::vector<size_t> shape = {
                    nested.size(), nested.empty() ? 0 : nested[0].size()};
                std::vector<T> flat;
                for (const auto &row : nested) {
                  if (row.size() != shape[1]) {
                    throw std::invalid_argument(
                        "Inconsistent row sizes in nested list");
                  }
                  flat.insert(flat.end(), row.begin(), row.end());
                }
                return TensorT(shape, std::move(flat));
              },
              py::arg("nested_list"))

          // ✅ versi 1D
          .def_static(
              "from_nested",
              [](const std::vector<T> &flat) {
                std::vector<size_t> shape = {flat.size()};
                return TensorT(shape, std::vector<T>(flat));
              },
              py::arg("flat_list"))

          .def(
              "data",
              [](TensorT &self) {
                return py::memoryview::from_buffer(
                    self.data(), {static_cast<ssize_t>(self.size())},
                    {static_cast<ssize_t>(sizeof(T))});
              },
              py::return_value_policy::reference_internal)
          // Cast ke berbagai tipe
          .def("cast_to_int8", &TensorT::template cast<int8_t>)
          .def("cast_to_uint8", &TensorT::template cast<uint8_t>)
          .def("cast_to_int16", &TensorT::template cast<int16_t>)
          .def("cast_to_uint16", &TensorT::template cast<uint16_t>)
          .def("cast_to_int32", &TensorT::template cast<int32_t>)
          .def("cast_to_uint32", &TensorT::template cast<uint32_t>)
          .def("cast_to_int64", &TensorT::template cast<int64_t>)
          .def("cast_to_uint64", &TensorT::template cast<uint64_t>)
          .def("cast_to_float32", &TensorT::template cast<float>)
          .def("cast_to_float64", &TensorT::template cast<double>)

          // reshape method
          .def("reshape", &TensorT::reshape, py::arg("new_shape"))

          // transpose() tanpa argumen
          .def("transpose",
               [](const TensorT &self) { return self.transpose(); })

          .def("flatten", [](const TensorT &self) { return self.flatten(); })
          .def(
              "add",
              [](const TensorT &self,
                 const TensorT &other) { return self.add(other); },
              py::arg("other"),
              "Add another tensor elementwise (shapes must match).")

          .def(
              "add_item",
              [](const TensorT &self, T value) { return self.add(value); },
              py::arg("value"), "Add scalar to all elements of tensor.")

          .def_static(
              "arange",
              [](T start, T stop,
                 T step) { return Tensor<T>::arange(start, stop, step); },
              py::arg("start"), py::arg("stop"), py::arg("step"),
              "Create a 1D tensor with values from start to stop (exclusive) "
              "with "
              "given step.")

          .def_static(
              "arange",
              [](T start, T stop,
                 T step) { return Tensor<T>::arange(start, stop, step); },
              py::arg("start"), py::arg("stop"), py::arg("step"),
              "Create a 1D tensor with values from start to stop (exclusive) "
              "with "
              "given step.")

          .def_static("zeros", &TensorT::zeros, py::arg("shape"));

  // Tambahkan khusus untuk tipe float
  if constexpr (std::is_same<T, float>::value) {
    cls.def_static("rand", &Tensor<float>::rand, py::arg("shape"),
                   py::arg("seed") = 0)
        .def_static("randn", &Tensor<float>::randn, py::arg("shape"),
                    py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f,
                    py::arg("seed") = 0)
        .def("matmul", [](const Tensor<float> &a, const Tensor<float> &b) {
          return a.matmul(b);
        });
  } else if constexpr (std::is_same<T, double>::value) {
    cls.def_static("rand", &Tensor<double>::rand, py::arg("shape"),
                   py::arg("seed") = 0)
        .def_static("randn", &Tensor<double>::randn, py::arg("shape"),
                    py::arg("mean") = 0.0, py::arg("stddev") = 1.0,
                    py::arg("seed") = 0)
        .def("matmul", [](const Tensor<double> &a, const Tensor<double> &b) {
          return a.matmul(b);
        });
  } else if constexpr (std::is_integral<T>::value) {
    cls.def_static("randint", &Tensor<T>::randint, py::arg("shape"),
                   py::arg("low") = 0, py::arg("high") = 100,
                   py::arg("seed") = 0)
        .def("matmul", [](const Tensor<T> &a, const Tensor<T> &b) {
          return a.matmul_int(b);
        });
  }
}

PYBIND11_MODULE(_larik, m) {
  bind_tensor<int8_t>(m, "int8");
  bind_tensor<uint8_t>(m, "uint8");
  bind_tensor<int16_t>(m, "int16");
  bind_tensor<uint16_t>(m, "uint16");
  bind_tensor<int32_t>(m, "int32");
  bind_tensor<uint32_t>(m, "uint32");
  bind_tensor<int64_t>(m, "int64");   // bisa juga int64_t
  bind_tensor<uint64_t>(m, "uint64"); // bisa juga int64_t
  bind_tensor<float>(m, "float32");
  bind_tensor<double>(m, "float64");
}
