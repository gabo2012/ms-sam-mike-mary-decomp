/* samunit: render one unit of a .spd voice on its own, so an encoding can be judged without the front end.
 *
 *   samunit voice.spd <unit> out.wav [--raw | --f0 HZ] [--dur MS]
 *
 * --raw plays the decoded frames at their stored pitch periods and native duration.
 * Otherwise it renders a segment at the requested flat pitch and duration.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sam.h"

static void put(FILE *f, unsigned v, int n)
{
    int i;
    for (i = 0; i < n; i++) fputc((int)((v >> (8 * i)) & 0xFF), f);
}

typedef struct {
    FILE *f;
    unsigned n;
} sink;

static int render_raw(sam_voice *v, int unit, sink *s)
{
    sam_unit u;
    float *mem;
    int noise_pos = 0, f, pos = 0;
    if (sam_unit_decode(v, unit, &noise_pos, &u) != 0) return -1;
    mem = calloc((size_t)u.order + 1, sizeof(float));
    if (!mem) {
        sam_unit_free(&u);
        return -1;
    }
    for (f = 0; f < u.nframes; f++) {
        int n = (int)fabsf(u.periods[f]), k, i;
        const float *a = u.lpc + f * (u.order + 1);
        for (k = 0; k < n; k++) {
            float sample = a[0] * u.exc[pos + k];
            for (i = u.order; i > 0; i--) {
                sample = (float)((double)sample - (double)mem[i] * a[i]);
                mem[i] = i > 1 ? mem[i - 1] : sample;
            }
            mem[0] = sample;
            if (sample > 32767.0f) sample = 32767.0f;
            else if (sample < -32768.0f) sample = -32768.0f;
            put(s->f, (uint16_t)(int16_t)(long)sample, 2);
        }
        pos += n;
        s->n += (unsigned)n;
    }
    free(mem);
    sam_unit_free(&u);
    return 0;
}

static void on_pcm(const int16_t *pcm, size_t n, void *user)
{
    sink *s = user;
    size_t i;
    for (i = 0; i < n; i++) put(s->f, (uint16_t)pcm[i], 2);
    s->n += (unsigned)n;
}

int main(int argc, char **argv)
{
    char err[256];
    sam_voice *v;
    sam_synth *sy;
    sam_params p;
    sam_segment seg;
    sink s;
    int i, unit, raw = 0, f0_set = 0, dur_set = 0;
    double f0 = 190.0, dur = 150.0;

    if (argc < 4) {
        fprintf(stderr, "usage: samunit voice.spd <unit> out.wav [--raw | --f0 HZ] [--dur MS]\n");
        return 1;
    }
    unit = atoi(argv[2]);
    for (i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--f0") && i + 1 < argc) { f0 = atof(argv[++i]); f0_set = 1; }
        else if (!strcmp(argv[i], "--dur") && i + 1 < argc) { dur = atof(argv[++i]); dur_set = 1; }
        else {
            fprintf(stderr, "samunit: unknown option %s\n", argv[i]);
            return 1;
        }
    }
    if (raw && (f0_set || dur_set)) {
        fprintf(stderr, "samunit: --raw cannot be combined with --f0 or --dur\n");
        return 1;
    }
    v = sam_voice_load(argv[1], err, sizeof err);
    if (!v) {
        fprintf(stderr, "samunit: %s\n", err);
        return 1;
    }
    sam_params_default(&p);
    s.f = fopen(argv[3], "wb");
    s.n = 0;
    if (!s.f) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    fwrite("RIFF\0\0\0\0WAVEfmt ", 1, 16, s.f);
    put(s.f, 16, 4), put(s.f, 1, 2), put(s.f, 1, 2);
    put(s.f, (unsigned)sam_voice_rate(v), 4), put(s.f, (unsigned)sam_voice_rate(v) * 2, 4), put(s.f, 2, 2), put(s.f, 16, 2);
    fwrite("data\0\0\0\0", 1, 8, s.f);
    if (raw) {
        if (render_raw(v, unit, &s)) {
            fprintf(stderr, "samunit: invalid unit or raw render failed\n");
            fclose(s.f);
            sam_voice_free(v);
            return 1;
        }
    } else {
        sy = sam_synth_new(v, &p);
        if (!sy) {
            fprintf(stderr, "samunit: out of memory\n");
            fclose(s.f);
            sam_voice_free(v);
            return 1;
        }
        memset(&seg, 0, sizeof seg);
        seg.unit = unit;
        seg.dur = (float)(dur / 1000.0);
        seg.n_knots = 2;
        seg.t[0] = 0.0f;
        seg.t[1] = (float)(dur / 1000.0 * sam_voice_rate(v));
        seg.f0[0] = seg.f0[1] = (float)f0;
        seg.amp[0] = seg.amp[1] = 1.0f;
        if (sam_synth_segment(sy, &seg, on_pcm, &s)) {
            fprintf(stderr, "samunit: render failed\n");
            fclose(s.f);
            sam_synth_free(sy);
            sam_voice_free(v);
            return 1;
        }
        sam_synth_free(sy);
    }
    fseek(s.f, 4, SEEK_SET);
    put(s.f, 36 + s.n * 2, 4);
    fseek(s.f, 40, SEEK_SET);
    put(s.f, s.n * 2, 4);
    fclose(s.f);
    printf("%s: unit %d of %d, %u samples at %d Hz%s\n", argv[3], unit, sam_voice_units(v), s.n,
           sam_voice_rate(v), raw ? " (raw pitch and native speed)" : "");
    sam_voice_free(v);
    return 0;
}
