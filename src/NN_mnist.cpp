#include "NN_mnist.h"

#include <iostream>
#include <fstream>
#include <assert.h>

namespace NN
{

MNIST::MNIST() :
  img_count(0),
  img_row(0),
  img_col(0)
{
}

MNIST::~MNIST()
{
}


void MNIST::LoadTrainData()
{ 
  loadData(
    "../data/train-images.idx3-ubyte",
    "../data/train-labels.idx1-ubyte",
    trainData);
}

void MNIST::LoadTestData()
{ 
  loadData(
    "../data/t10k-images.idx3-ubyte", 
    "../data/t10k-labels.idx1-ubyte", 
    testData);
}

void MNIST::loadData(const char* data_path, const char* label_path, std::vector<Data>& dst)
{
  // image
  {
    std::ifstream ifs_img(data_path, std::ios::in | std::ios::binary);
    assert(!ifs_img.fail());
    int ival = 0;
    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_count = ival;

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_row = ival;

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_col = ival;

    trainData.reserve(img_count);


    // label
    unsigned char* labels = new unsigned char[img_count];
    {
      std::ifstream ifs_lbl(label_path, std::ios::in | std::ios::binary);

      assert(!ifs_lbl.fail());
      int ival = 0;
      ifs_lbl.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);

      ifs_lbl.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      assert(img_count == ival);
      ifs_lbl.read((char*)labels, sizeof(labels[0])*img_count);
      ifs_lbl.close();
    }

    float tbl[10][10] =
    {
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    };

    // each image
    const size_t sz = img_row * img_col;
    unsigned char* bt = new unsigned char[sz];
    float* fl = new float[sz];
    for (int di = 0; di < img_count; ++di) {
      ifs_img.read((char*)bt, sizeof(unsigned char) * sz);
      for (int i = 0; i < sz; ++i) {
        fl[i] = bt[i] / 255.0f;
      }
      int d = (int)labels[di];
      dst.push_back( Data(fl, tbl[d]) );
    }
    delete[] fl;
    delete[] bt;
  }
}


const Matrix& MNIST::GetTestInputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT((DataSize + 1) == mat->col());
  _ASSERT((idx + count) <= testData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < DataSize; ++j) {
      (*mat)(x, j) = testData[i].input[j];
    }
    (*mat)(x, DataSize) = 1.0f;
  }
  return *mat;
}
const Matrix& MNIST::GetTestOutputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT(LabelSize == mat->col());
  _ASSERT((idx + count) <= testData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < LabelSize; ++j) {
      (*mat)(x, j) = testData[i].out[j];
    }
  }
  return *mat;
}


}