#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Error, please specify the folder containing the benchmark data!\n");
        return -1;
    }
    char* bench_dir = argv[1];

    DIR* d;
    struct dirent* dir;
    d = opendir(bench_dir);
    if (!d) {
        printf("Error opening folder \"%s\"!\n", bench_dir);
        return -1;
    }

    char* gnu_commands[6];
    gnu_commands[0] = "set title \"GEMM Performance\" font \",14\"";
    gnu_commands[1] = "set key right bottom";
    gnu_commands[2] = "set grid";
    gnu_commands[3] = "set ylabel \"GFLOPS\" font \",11\"";
    gnu_commands[4] = "set xlabel \"m=n=k\" font \",11\"";
    gnu_commands[5] = "";

    int i = 5;
    char buffer[10000];
    strcpy(buffer, "plot ");
    while ((dir = readdir(d)) != NULL) {
        if (dir->d_type == DT_REG) {
            strcat(buffer, "\"");
            strcat(buffer, bench_dir);
            strcat(buffer, "/");
            strcat(buffer, dir->d_name);
            strcat(buffer, "\" using 1:2 title \"");
            strcat(buffer, dir->d_name);
            strcat(buffer, "\" with lines lw 2, ");
        }
    }
    gnu_commands[i] = buffer;
    closedir(d);

    FILE* gnupipe = NULL;
    gnupipe = popen("gnuplot -persistent", "w");
    for (int i = 0; i < 6; i++) {
        fprintf(gnupipe, "%s\n", gnu_commands[i]);
    }
    return 0;
}