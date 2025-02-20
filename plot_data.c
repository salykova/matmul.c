#include <stdio.h>

int main() {
    FILE* gnupipe = NULL;
    char* gnu_commands[] = {
        "set title \"Median GFLOPS\" font \",14\"",
        "set key right bottom",
        "set ylabel \"GFLOPS\" font \",11\"",
        "set xlabel \"m=n=k\" font \",11\"",
        "plot \"benchmark_matmul.txt\" using 1:3 title \"matmul.c\" with lines lw 2, "
        "\"benchmark_openblas.txt\" using 1:3 title \"OpenBLAS\" with lines lw 2"};
    int n = 5;
    gnupipe = popen("gnuplot -persistent", "w");
    for (int i = 0; i < n; i++) {
        fprintf(gnupipe, "%s\n", gnu_commands[i]);
    }
    return 0;
}