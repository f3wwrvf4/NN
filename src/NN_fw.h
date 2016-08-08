#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>

#include <NN_net.h>

namespace NN
{
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
template <class CONTENT>
void Train(CONTENT& data, const char* fpath,
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
          data.GetTrainOutputData(idx, batch_size, &out));
      }
      net.save(fpath);
    }
  }
}


template <class CONTENT>
void Test(const CONTENT& data, const char* fpath,
  NN::Network::InitParam* init_param, int layer_num)
{
  NN::Network net(layer_num, init_param, 1);
  net.load(fpath);

  NN::Matrix in(1, CONTENT::DataSize);
  NN::Matrix out(1, CONTENT::LabelSize);
  const int LabelSize = CONTENT::LabelSize;

  int count = data.GetTestDataCount();
  int answer[LabelSize][LabelSize] = {};

  for (int i = 0; i < count; ++i) {
    data.GetTestInputData(i, 1, &in);
    data.GetTestOutputData(i, 1, &out);

    const NN::Matrix& res = net.eval(in);

    float x[LabelSize];
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