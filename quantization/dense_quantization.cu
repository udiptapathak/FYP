#include <unordered_map>
#include <up/primitive.h>

static const u16 BUCKET_COUNT = 8096;
const static f32 BUCKET_COUNT_F32 = 8096.l;
const static int DIAMETER = 10;
const static u8 MASK_128 = 1 << 7;


typedef struct {
	f32 lim;
	u8 shouldUpdate;
	f32 base[3];
} Qmap;

static inline __device__ int getBucketIndex(f32 lim, f32 weight) {
	return (int) ((weight - lim) / BUCKET_COUNT_F32) + 1;
}

static inline __device__ f32 getVariance(f32 mean, f32 sum, f32 sumSq,
	const u32 N)
{
	return mean * mean + (2 * mean * sum + sum * sum) / N;
}

std::unordered_map<u8, Qmap> qmaps;

extern "C" {
Qmap QmapNew(f32 inDim)
{
	const f32 lim = 1 / sqrt(inDim); 
	return (Qmap) {lim, 0, {0.l, -lim, lim}};
}



__global__ void QmapApply(Qmap qmap, vp weight_vp, const u32 N)
{
	f32p weight = (f32p) weight_vp;
	u8 cur = 0;
	int node = 0;
	f32p base = qmap.base;
	f32 t_f32;
	const u8 DEPTH = 1;
	for (u32 i = 0 ; i < N ; i++) {
		t_f32 = weight[i];
		while (cur < DEPTH) {
			cur++;
			node = node << 1;
			node += base[node] < t_f32 ? 1 : 2;
		}
		weight[i] = base[node];
	}
}

__global__ void QmapUpdate(Qmap qmap, vp weight_vp, const u32 N)
{
	f32 limLo = - qmap.lim;
	f32 limHi = qmap.lim;
	f32p weight = (f32p) weight_vp;
	u32 bucket[BUCKET_COUNT];
	for (u32 i = 0 ; i < BUCKET_COUNT ; i++)
		bucket[i] = 0;
	for (u32 i = 0 ; i < N ; i++)
		bucket[getBucketIndex(limLo, weight[i])]++;
	u64 sum = 0.l;
	u64 sumSq = 0.l;
	const f32 BUCKET_SIZE= (limHi - limLo) / BUCKET_COUNT;
	f32 t_f32 = limLo - BUCKET_SIZE / 2;
	f32 tb_f32;
	for (u32 i = 0 ; i < BUCKET_COUNT ; i++) {
		t_f32 += BUCKET_SIZE;
		tb_f32 = t_f32 * bucket[i];
		sum += tb_f32;
		sumSq += tb_f32 * tb_f32;
	}
	f32 cur = 0.l;
	f32 curSq = 0.l;
	f32 mean1, mean2, var1, var2;
	f32 minVar = F32_MAX;
	u32 limPoint = 0;
	u32 t1_u32 = 0;
	u32 t2_u32 = 0;
	f32 limMean1, limMean2;
	t_f32 = limLo - BUCKET_SIZE / 2;
	for (u32 i = 0 ; i < BUCKET_COUNT ; i++) {
		t_f32 += BUCKET_SIZE;
		t2_u32 = bucket[i];
		t1_u32 += t2_u32;
		t_f32 = t_f32 * t2_u32;
		cur += tb_f32;
		curSq += tb_f32 * tb_f32;
		mean1 = cur / t1_u32;
		mean2 = (sum - cur) / (N - t1_u32);
		var1 = getVariance(mean1, cur, curSq, t1_u32);
		var2 = getVariance(mean2, sum - cur, sumSq - curSq, N - t1_u32);
		t_f32 = var1 + var2;
		if (t_f32 < minVar) {
			minVar = t_f32;
			limPoint = i;
			limMean1 = mean1;
			limMean2 = mean2;
		}
	}
	qmap.base[0] = weight[limPoint] + F32_MIN;
	qmap.base[1] = limMean1;
	qmap.base[2] = limMean2;
}

bool QmapShouldUpdate(Qmap qmap)
{
	if (qmap.shouldUpdate & MASK_128) {
		qmap.shouldUpdate = 0;
		return true;
	}
	qmap.shouldUpdate++;
	return false;
}

#include <stdio.h>
Qmap QmapLayer(vp layer, u32 inDim)
{
	printf("%p<<\n", layer);
	printf("%u<<\n", (u32) inDim);
	fflush(stdout);
	Qmap ret;
	return ret;
	// auto qmap = qmaps.find(layer);
	// if (qmap != qmaps.end())
	// 	return qmap->second;
	// return qmaps[layer] = QmapNew(inDim);
}

}
