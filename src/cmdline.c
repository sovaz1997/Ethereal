/*
  Ethereal is a UCI chess playing engine authored by Andrew Grant.
  <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>

  Ethereal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Ethereal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitboards.h"
#include "board.h"
#include "cmdline.h"
#include "evaluate.h"
#include "move.h"
#include "search.h"
#include "thread.h"
#include "time.h"
#include "transposition.h"
#include "tuner.h"
#include "uci.h"

void handleCommandLine(int argc, char **argv) {

    // Benchmarker is being run from the command line
    // USAGE: ./Ethereal bench <depth> <threads> <hash>
    if (argc > 1 && strEquals(argv[1], "bench")) {
        runBenchmark(argc, argv);
        exit(EXIT_SUCCESS);
    }

    //
    if (argc > 1 && strEquals(argv[1], "filter")) {
        filterBook(argv[2]);
        exit(EXIT_SUCCESS);
    }

    //
    if (argc > 1 && strEquals(argv[1], "nnbook")) {
        buildNNBook(argv[2]);
        exit(EXIT_SUCCESS);
    }

    // Bench is being run from the command line
    // USAGE: ./Ethereal evalbook <book> <depth> <threads> <hash>
    if (argc > 2 && strEquals(argv[1], "evalbook")) {
        runEvalBook(argc, argv);
        exit(EXIT_SUCCESS);
    }

    // Tuner is being run from the command line
    #ifdef TUNE
        runTuner();
        exit(EXIT_SUCCESS);
    #endif
}

void runBenchmark(int argc, char **argv) {

    static const char *Benchmarks[] = {
        #include "bench.csv"
        ""
    };

    Board board;
    Thread *threads;
    Limits limits = {0};

    int scores[256];
    double times[256];
    uint64_t nodes[256];
    uint16_t bestMoves[256];
    uint16_t ponderMoves[256];

    double time;
    uint64_t totalNodes = 0ull;

    int depth     = argc > 2 ? atoi(argv[2]) : 13;
    int nthreads  = argc > 3 ? atoi(argv[3]) :  1;
    int megabytes = argc > 4 ? atoi(argv[4]) : 16;

    initTT(megabytes);
    time = getRealTime();
    threads = createThreadPool(nthreads);

    // Initialize a "go depth <x>" search
    limits.multiPV        = 1;
    limits.limitedByDepth = 1;
    limits.depthLimit     = depth;

    for (int i = 0; strcmp(Benchmarks[i], ""); i++) {

        // Perform the search on the position
        limits.start = getRealTime();
        boardFromFEN(&board, Benchmarks[i], 0);
        getBestMove(threads, &board, &limits, &bestMoves[i], &ponderMoves[i]);

        // Stat collection for later printing
        scores[i] = threads->info->values[depth];
        times[i] = getRealTime() - limits.start;
        nodes[i] = nodesSearchedThreadPool(threads);

        clearTT(); // Reset TT between searches
    }

    printf("\n=================================================================================\n");

    for (int i = 0; strcmp(Benchmarks[i], ""); i++) {

        // Convert moves to typical UCI notation
        char bestStr[6], ponderStr[6];
        moveToString(bestMoves[i], bestStr, 0);
        moveToString(ponderMoves[i], ponderStr, 0);

        // Log all collected information for the current position
        printf("Bench [# %2d] %5d cp  Best:%6s  Ponder:%6s %12d nodes %8d nps\n", i + 1, scores[i],
            bestStr, ponderStr, (int)nodes[i], (int)(1000.0f * nodes[i] / (times[i] + 1)));
    }

    printf("=================================================================================\n");

    // Report the overall statistics
    time = getRealTime() - time;
    for (int i = 0; strcmp(Benchmarks[i], ""); i++) totalNodes += nodes[i];
    printf("OVERALL: %53d nodes %8d nps\n", (int)totalNodes, (int)(1000.0f * totalNodes / (time + 1)));

    free(threads);
}

void runEvalBook(int argc, char **argv) {

    Board board;
    char line[256];
    Limits limits = {0};
    uint16_t best, ponder;
    double start = getRealTime();

    FILE *book    = fopen(argv[2], "r");
    int depth     = argc > 3 ? atoi(argv[3]) : 12;
    int nthreads  = argc > 4 ? atoi(argv[4]) :  1;
    int megabytes = argc > 5 ? atoi(argv[5]) :  2;

    Thread *threads = createThreadPool(nthreads);

    limits.multiPV = 1;
    limits.limitedByDepth = 1;
    limits.depthLimit = depth;
    initTT(megabytes);

    while ((fgets(line, 256, book)) != NULL) {
        limits.start = getRealTime();
        boardFromFEN(&board, line, 0);
        getBestMove(threads, &board, &limits, &best, &ponder);
        resetThreadPool(threads); clearTT();
        printf("FEN: %s", line);
    }

    printf("Time %dms\n", (int)(getRealTime() - start));
}

void filterBook(char *fname) {

    char line[256];
    FILE *fin = fopen(fname, "r");

    Thread *thread = createThreadPool(1);
    Board *board   = &thread->board;

    while (1) {

        if (fgets(line, 256, fin) == NULL)
            break;

        boardFromFEN(board, line, 0);

        // Remove all in-check positions
        if (board->kingAttackers)
            continue;

        // Remove all Tablebase positions
        if (popcount(board->colours[WHITE] | board->colours[BLACK]) <= 6)
            continue;

        // Remove positions where qs and eval differ
        int ev = evaluateBoard(thread, board);
        int qs = qsearch(thread, &thread->pv, -MATE, MATE);
        if (ev != qs) continue;

        printf("%s", line);
    }

    free(thread);
}

void buildNNBook(char *fname) {

    #define encode_piece(p) (8 * pieceColour(p) + pieceType(p))
    #define pack_pieces(p1, p2) (((p1) << 4) | (p2))

    char line[256];
    FILE *fin = fopen(fname, "r");
    FILE *fout = fopen("output.nnbook", "wb");

    Thread *thread = createThreadPool(1);
    Board *board   = &thread->board;

    while (fgets(line, 256, fin) != NULL) {

        boardFromFEN(board, line, 0);

        char *tail   = strstr(line, "] ");
        int16_t eval = atoi(tail + strlen("] "));
        int8_t turn  = board->turn;

        uint64_t white  = board->colours[WHITE];
        uint64_t black  = board->colours[BLACK];
        uint64_t pieces = white | black;

        uint8_t count = popcount(pieces);
        uint8_t wksq  = getlsb(white & board->pieces[KING]);
        uint8_t bksq  = getlsb(black & board->pieces[KING]);

        uint8_t types[32] = {0};
        uint8_t packed[16] = {0};

        uint8_t result;
        if (strstr(line, "[0.0]")) result = 0u;
        if (strstr(line, "[0.5]")) result = 1u;
        if (strstr(line, "[1.0]")) result = 2u;

        fwrite(&pieces, sizeof(uint64_t), 1, fout);
        fwrite(&eval,   sizeof(int16_t ), 1, fout);
        fwrite(&result, sizeof(uint8_t ), 1, fout);
        fwrite(&turn,   sizeof(uint8_t ), 1, fout);
        fwrite(&wksq,   sizeof(uint8_t ), 1, fout);
        fwrite(&bksq,   sizeof(uint8_t ), 1, fout);
        fwrite(&count,  sizeof(uint8_t ), 1, fout);

        for (int i = 0; pieces; i++) {
            int sq = poplsb(&pieces);
            types[i] = encode_piece(board->squares[sq]);
        }

        for (int i = 0; i < 16; i++)
            packed[i] = pack_pieces(types[i*2], types[i*2+1]);

        fwrite(packed, sizeof(uint8_t), (count + 1) / 2, fout);
    }

    fclose(fin);
    fclose(fout);
    free(thread);

    #undef encode_piece
    #undef pack_pieces
}