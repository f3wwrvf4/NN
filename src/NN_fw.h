#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <NN_net.h>

namespace NN
{
template <int InSize, int OutSize>
class ContentData
{
public:
  static const int DataSize = InSize;
  static const int LabelSize = OutSize;

  int dataCount() const { return (int)data.size(); }
  int inputSize() const { return  DataSize; }
  int outputSize() const { return  LabelSize; }
  void reserve(int size) { data.reserve(size); }
  void push_back(float in[InSize], float out[OutSize]) { data.push_back(Data(in, out)); }
  void shuffle() { random_shuffle(data.begin(), data.end()); }
  const Matrix& inputData(int idx, int count, Matrix& mat) const
  {
    _ASSERT(count == mat.row());
    _ASSERT((DataSize) == mat.col());
    _ASSERT((idx + count) <= data.size());

    for (int i = idx; i < idx + count; ++i) {
      int x = i%count;
      for (int j = 0; j < DataSize; ++j) {
        mat(x, j) = data[i].input[j];
      }
    }
    return mat;
  }
  const Matrix& outputData(int idx, int count, Matrix& mat) const
  {
    _ASSERT(count == mat.row());
    _ASSERT(LabelSize == mat.col());
    _ASSERT((idx + count) <= data.size());

    for (int i = idx; i < idx + count; ++i) {
      int x = i%count;
      for (int j = 0; j < LabelSize; ++j) {
        mat(x, j) = data[i].out[j];
      }
    }
    return mat;
  }

private:
  struct Data
  {
    Data(float in[DataSize], float ou[LabelSize])
    {
      for (int i = 0; i < DataSize; ++i) {
        input[i] = in[i];
      }
      for (int i = 0; i < LabelSize; ++i) {
        out[i] = ou[i];
      }
    }
    float input[DataSize];
    float out[LabelSize];
  };
  std::vector<Data> data;
};

#if 0
template <class CONTENT>
void AutoEncode(CONTENT& data, const char* fpath,
  NN::Network::InitParam* init_param, int layer_num,
  int batch_size, int train_count)
{
  {
    int dataCount = data.GetTrainDataCount();

    NN::Network net(layer_num, init_param, batch_size);
    net.load(fpath);

    NN::Matrix in(batch_size, CONTENT::DataSize);
    NN::Matrix out(batch_size, CONTENT::LabelSize);
    const int count = dataCount / batch_size;

    while (train_count--) {
      data.shuffle();
      for (int i = 0; i < count; ++i) {
        int idx = i*batch_size;
        net.train(
          data.GetTrainInputData(idx, batch_size, &in),
          data.GetTrainInputData(idx, batch_size, &in),      // since its autoencode
      }
      net.save(fpath);
    }
  }
}
#endif

template <class Content>
void Train(NN::Network& net, Content& data,
  int batch_size, int train_count)
{
  int dataCount = data.dataCount();

  NN::Matrix in(batch_size, data.inputSize());
  NN::Matrix out(batch_size, data.outputSize());
  const int count = dataCount / batch_size;

  while (train_count--) {
    data.shuffle();
    for (int i = 0; i < count; ++i) {
      int idx = i*batch_size;
      net.train(data.inputData(idx, batch_size, in), data.outputData(idx, batch_size, out));
    }
  }
}

template <class Content>
void Test(const NN::Network& net, const Content& data)
{
  NN::Matrix in(1, data.inputSize());
  NN::Matrix out(1, data.outputSize());
  const int LabelSize = data.outputSize();

  int count = data.dataCount();
  int answer[Content::LabelSize][Content::LabelSize] = {};

  for (int i = 0; i < count; ++i) {
    data.inputData(i, 1, in);
    data.outputData(i, 1, out);

    const NN::Matrix& res = net.eval(in);

    float x[Content::LabelSize];
    for (int j = 0; j < LabelSize; ++j) {
      x[j] = 1.0f - res(0, j);
      x[j] *= x[j];
    }

    float t = x[0];
    int min = 0;
    for (int j = 1; j < LabelSize; ++j) {
      if (x[j] < t) {
        min = j;
        t = x[j];
      }
    }

    int actual = 0;
    for (int j = 1; j < LabelSize; ++j) {
      if (out(0, j) != 0) {
        actual = j;
        break;
      }
    }

    ++answer[actual][min];
  }

  for (int i = 0; i < LabelSize; ++i) {
    std::cout << "[" << i << "] ";
    for (int j = 0; j < LabelSize; ++j) {
      std::cout << std::right << std::setw(6) << answer[i][j] << " ";
    }
    int a_sam = 0;
    for (int j = 0; j < LabelSize; ++j) {
      a_sam += answer[i][j];
    }
    const int a = answer[i][i];
    const int b = a_sam;
    const float per = a / (float)b;
    std::cout << "(" <<
      std::setw(4) << a << "/" <<
      std::setw(4) << b << ")  " <<
      std::setw(4) << per * 100 << "%" <<
      std::endl;
  }
  {
    int correct = 0;
    for (int j = 0; j < LabelSize; ++j) {
      correct += answer[j][j];
    }
    const int a = correct;
    const int b = count;
    const float per = a / (float)b;

    std::cout << "all: (" <<
      std::setw(4) << a << "/" <<
      std::setw(4) << b << ")  " <<
      std::setw(4) << per * 100 << "%" <<
      std::endl;
  }
}

}