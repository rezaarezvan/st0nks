// Header files
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>

float calculate_volatility(float spot_price, int time_steps) {
  const char *file_name = "data.csv";
  std::ifstream file_ptr;
  file_ptr.open(file_name, std::ifstream::in);
  if (!file_ptr.is_open()) {
    std::cerr << "Cannot open data.csv! Exiting..\n";
    exit(EXIT_FAILURE);
  }

  std::string line;
  if (!std::getline(file_ptr, line)) {
    std::cerr << "Cannot read from data.csv! Exiting..\n";
    file_ptr.close();
    exit(EXIT_FAILURE);
  }
  file_ptr.close();

  int i = 0, len = time_steps - 1;
  std::unique_ptr<float[]> priceArr = std::make_unique<float[]>(time_steps - 1);
  std::istringstream iss(line);
  std::string token;

  while (std::getline(iss, token, ','))
    priceArr[i++] = std::stof(token);

  float sum = spot_price;

  for (i = 0; i < len; i++)
    sum += priceArr[i];
  float mean_price = sum / (len + 1);

  sum = powf((spot_price - mean_price), 2.0f);
  for (i = 0; i < len; i++)
    sum += powf((priceArr[i] - mean_price), 2.0f);

  float std_dev = sqrtf(sum);

  return std_dev / 100.0f;
}

float *find_2d_mean(float **matrix, int num_loops, int time_steps) {
  int j;
  float *avg = new float[time_steps];
  float sum = 0.0f;

  for (int i = 0; i < time_steps; i++) {

#pragma omp parallel for private(j) reduction(+ : sum)
    for (j = 0; j < num_loops; j++) {
      sum += matrix[j][i];
    }

    avg[i] = sum / num_loops;
    sum = 0.0f;
  }

  return avg;
}

float rand_gen(float mean, float std_dev) {
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(static_cast<unsigned int>(seed));
  std::normal_distribution<float> distribution(mean, std_dev);
  return distribution(generator);
}

float *run_black_scholes_model(float spot_price, int time_steps,
                               float risk_rate, float volatility) {
  float mean = 0.0f, std_dev = 1.0f;
  float deltaT = 1.0f / time_steps;
  std::unique_ptr<float[]> norm_rand =
      std::make_unique<float[]>(time_steps - 1);
  float *stock_price = new float[time_steps];
  stock_price[0] = spot_price;

  for (int i = 0; i < time_steps - 1; i++)
    norm_rand[i] = rand_gen(mean, std_dev);

  for (int i = 0; i < time_steps - 1; i++)
    stock_price[i + 1] =
        stock_price[i] *
        exp(((risk_rate - (powf(volatility, 2.0f) / 2.0f)) * deltaT) +
            (volatility * norm_rand[i] * sqrtf(deltaT)));

  return stock_price;
}

int main(int argc, char **argv) {
  clock_t t = clock();

  int in_loops = 100;
  int out_loops = 10000;
  int time_steps = 180;

  float **stock = new float *[in_loops];
  for (int i = 0; i < in_loops; i++)
    stock[i] = new float[time_steps];

  float **avg_stock = new float *[out_loops];
  for (int i = 0; i < out_loops; i++)
    avg_stock[i] = new float[time_steps];

  float *opt_stock = new float[time_steps];

  float risk_rate = 0.001f;
  float spot_price = 100.0f;

  float volatility = calculate_volatility(spot_price, time_steps);

  std::cout << "--Welcome to Stockast: Stock Forecasting Tool--\n";
  std::cout << "  Using market volatility = " << volatility << std::endl;

  int i;

#pragma omp parallel private(i)
  {
#pragma omp single
    {
      int numThreads = omp_get_num_threads(); // Number of threads
      std::cout << "  Using " << numThreads << " thread(s)\n\n";
      std::cout << "  Have patience! Computing..";
      omp_set_num_threads(numThreads);
    }

#pragma omp for schedule(dynamic)
    for (i = 0; i < out_loops; i++) {
      for (int j = 0; j < in_loops; j++)
        stock[j] = run_black_scholes_model(spot_price, time_steps, risk_rate,
                                           volatility);

      avg_stock[i] = find_2d_mean(stock, in_loops, time_steps);
    }
  }

  opt_stock = find_2d_mean(avg_stock, out_loops, time_steps);

  std::ofstream file_ptr;
  file_ptr.open("opt.csv", std::ofstream::out);
  if (!file_ptr.is_open()) {
    std::cerr << "Couldn't open opt.csv! Exiting..\n";
    return EXIT_FAILURE;
  }

  for (i = 0; i < time_steps; i++)
    file_ptr << opt_stock[i] << "\n";
  file_ptr.close();

  for (i = 0; i < in_loops; i++)
    delete[] stock[i];
  delete[] stock;

  for (i = 0; i < out_loops; i++)
    delete[] avg_stock[i];
  delete[] avg_stock;

  delete[] opt_stock;

  t = clock() - t;
  std::cout << " done!\n  Time taken = "
            << static_cast<float>(t / CLOCKS_PER_SEC) << "s";

  return getchar();
}
