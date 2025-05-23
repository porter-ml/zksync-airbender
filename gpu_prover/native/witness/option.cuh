#pragma once

#include "common.cuh"

enum OptionTag : u32 {
  None,
  Some,
};

template <typename T> struct Option {
  OptionTag tag;
  T value;
};
