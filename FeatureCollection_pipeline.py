import numpy as np
import pandas as pd
import multiprocessing
import logging
import random
import pysam  # <-- 新增：用于读取参考基因组长度
import time  # 添加计时功能

from Depth_Methy_Features import DepthCalculator, MethylationProcessor  # 确保导入正确的模块

def get_chrom_lengths_from_fasta(reference_file):
    """
    从 reference_file (fasta) 中获取所有染色体的长度，返回字典:
    {
        '1': 248956422,
        '2': 242193529,
        ...
    }
    """
    with pysam.FastaFile(reference_file) as ref_f:
        chrom_lengths = {}
        for chrom in ref_f.references:
            length = ref_f.get_reference_length(chrom)
            # 一般 fasta 中的"chrom"名称里还包含 "chr" 前缀，
            # 如果你的 CNV bed 里不带 "chr"，可自行去掉前缀:
            stripped_chrom = chrom.replace("chr", "")
            chrom_lengths[stripped_chrom] = length
    return chrom_lengths

class TrainingDataBuilderRegion:
    """
    针对"直接在 process_chromosome 阶段指定区段"的示例：
    1. 在每个染色体找到 CNV ±10kb 作为 nearCNV 区段
    2. 从余下区域抽取作为 farCNV 区段，并进行更严格的下采样(仅保留20%)
    3. 调用 DepthCalculator & MethylationProcessor 只在这些区段做计算
    4. 贴 0/1/2 标签并返回
    """
    def __init__(self, bin_size=100, context_size=10000, normal_keep_fraction=0.0, n_processes=None):
        """
        Parameters
        ----------
        bin_size : int
            bin大小
        context_size : int
            对CNV区域前后扩展的上下文大小
        normal_keep_fraction : float
            对远离CNV的正常区段随机保留的比例(例如0.2=20%)
        n_processes : int or None
            多进程数
        """
        self.bin_size = bin_size
        self.context_size = context_size
        self.normal_keep_fraction = normal_keep_fraction
        self.n_processes = n_processes
        self.logger = logging.getLogger("lstm_training")

    def process_chromosome(self, chrom, bam_file, meth_file, reference_file, chrom_cnv_df, chrom_length=None):
        """
        仅对"CNV ± context"以及抽样选出的"远离CNV区段"做特征提取，贴标签 (0/1/2)。
        chrom_cnv_df : 仅包含该染色体的CNV行
        chrom_length : 该染色体长度(若需确定远离CNV区段范围，如LD region等)
        """
        start_time = time.time()
        self.logger.info(f"开始处理染色体 {chrom}")
        
        if len(chrom_cnv_df) == 0:
            self.logger.info(f"[{chrom}] No CNVs found, skipping.")
            return None

        # 1) 构建 nearCNV 区段 (CNV ± context_size)
        intervals = []
        for _, row in chrom_cnv_df.iterrows():
            start_ctx = max(0, row['Start'] - self.context_size)
            end_ctx   = row['End'] + self.context_size
            intervals.append((start_ctx, end_ctx))

        # 合并重叠区段
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_start, cur_end = intervals[0]
        for i in range(1, len(intervals)):
            st, ed = intervals[i]
            if st <= cur_end:
                cur_end = max(cur_end, ed)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = st, ed
        merged.append((cur_start, cur_end))

        # 日志：输出 nearCNV 区段个数
        self.logger.info(f"[{chrom}] Merged nearCNV intervals count = {len(merged)}")

        # 2) 构建"远离CNV"的正常区段(示例)
        far_regions = []
        if chrom_length is None:
            # 若不提供染色体长度，则跳过"远离CNV"区段
            self.logger.warning(f"[{chrom}] chrom_length not provided, skip far-region.")
        else:
            # 将 [0, chrom_length) 区段去掉 merged 部分 => 得到剩余 gap
            prev_end = 0
            for (st, ed) in merged:
                if st > prev_end:
                    far_regions.append((prev_end, st))
                prev_end = ed
            if prev_end < chrom_length:
                far_regions.append((prev_end, chrom_length))

            # 随机丢弃绝大部分far-region，仅保留 normal_keep_fraction => 0.2
            keep_num = int(len(far_regions) * self.normal_keep_fraction)
            keep_regions = random.sample(far_regions, keep_num) if keep_num > 0 else []
            far_regions = sorted(keep_regions, key=lambda x: x[0])

            # 日志：输出 far_regions 原始数量和下采样后数量
            self.logger.info(f"[{chrom}] far_regions raw: {len(far_regions)} -> downsampled to {len(keep_regions)}")

        # 将 nearCNV + far_regions 合并 => final_intervals
        final_intervals = merged + far_regions
        self.logger.info(f"[{chrom}] final intervals count = {len(final_intervals)}")

        # 3) 首先创建全局MethylationProcessor，只加载一次数据
        try:
            self.logger.info(f"[{chrom}] 创建全局MethylationProcessor对象")
            global_meth_start = time.time()
            global_meth_processor = MethylationProcessor(
                meth_file=meth_file,
                reference_filename=reference_file,
                bin_size=self.bin_size,
                region=None  # 加载所有数据供后续筛选使用
            )
            self.logger.info(f"[{chrom}] 全局MethylationProcessor创建完成，用时 {time.time() - global_meth_start:.2f} 秒")
        except Exception as e:
            self.logger.error(f"[{chrom}] 创建MethylationProcessor失败: {str(e)}")
            return None
            
        # 4) 分段调用 DepthCalculator 并贴标签
        all_sub_dfs = []
        all_sub_labels = []

        # 处理各个区间
        for idx, (int_start, int_end) in enumerate(final_intervals):
            interval_start_time = time.time()
            self.logger.info(f"[{chrom}] 处理区间 {idx+1}/{len(final_intervals)}: {int_start}-{int_end}")
            
            if int_end <= int_start:
                self.logger.warning(f"[{chrom}] 跳过无效区间: {int_start}-{int_end}")
                continue  # 避免异常区段
                
            try:
                # 将global_meth_processor传递给DepthCalculator，让它只提取当前区域的数据
                depth_calculator = DepthCalculator(
                    filename=bam_file,
                    reference_filename=reference_file,
                    bin_size=self.bin_size,
                    region=(int_start, int_end),  # 仅处理这段
                    methylation_processor=global_meth_processor  # 传入全局对象，但会自动只提取当前区域数据
                )
                depth_feats = depth_calculator.process_region(chrom)
                
                # 关闭depth_calculator以释放资源
                depth_calculator.close()

                # 区域甲基化处理 - 单独处理这个区间的甲基化特征
                region_meth_processor = MethylationProcessor(
                    meth_file=meth_file,
                    reference_filename=reference_file,
                    bin_size=self.bin_size,
                    region=(int_start, int_end)
                )
                meth_feats = region_meth_processor.process_chromosome(chrom)

                if depth_feats is None or meth_feats is None:
                    self.logger.warning(f"[{chrom}] 区间 {int_start}-{int_end} 未能获取特征，跳过")
                    continue

                # 将meth_feats中的内容合并到depth_feats
                if meth_feats is not None:
                    for key, value in meth_feats.items():
                        if key not in ['bin_coords', 'bin_start', 'chrom'] and len(value) == len(depth_feats['rd_u']):
                            depth_feats[key] = value

                # 创建特征数据框
                sub_df = pd.DataFrame({
                    'rd_u': depth_feats['rd_u'],
                    'gc_content': depth_feats['gc_content'],
                    'methyl_penalty': depth_feats['methyl_penalty'],
                    # 'meth_density': depth_feats['meth_density'],
                    # 'meth_variance': depth_feats['meth_variance'],
                    'chrom': chrom
                })
                
                # 添加深度-甲基化比例特征
                if 'depth_var_ratio' in depth_feats:
                    sub_df['depth_var_ratio'] = depth_feats['depth_var_ratio']
                if 'depth_density_ratio' in depth_feats:
                    sub_df['depth_density_ratio'] = depth_feats['depth_density_ratio']
                if 'penalty_var_ratio' in depth_feats:
                    sub_df['penalty_var_ratio'] = depth_feats['penalty_var_ratio']

                # 添加局部上下文特征
                if 'depth_gradient' in depth_feats:
                    sub_df['depth_gradient'] = depth_feats['depth_gradient']
                if 'depth_smooth' in depth_feats:
                    sub_df['depth_smooth'] = depth_feats['depth_smooth']
                if 'depth_local_std' in depth_feats:
                    sub_df['depth_local_std'] = depth_feats['depth_local_std']
                if 'depth_change_score' in depth_feats:
                    sub_df['depth_change_score'] = depth_feats['depth_change_score']

                # 添加甲基化局部上下文特征
                if 'meth_var_gradient' in depth_feats:
                    sub_df['meth_var_gradient'] = depth_feats['meth_var_gradient']
                if 'meth_var_smooth' in depth_feats:
                    sub_df['meth_var_smooth'] = depth_feats['meth_var_smooth']
                if 'depth_meth_coherence' in depth_feats:
                    sub_df['depth_meth_coherence'] = depth_feats['depth_meth_coherence']

                # 标记每个bin的CNV类型
                N_bins = len(sub_df)
                labels = np.zeros(N_bins, dtype=int)
                for bin_idx in range(N_bins):
                    bin_global_start = int_start + bin_idx*self.bin_size
                    bin_global_end   = bin_global_start + self.bin_size
                    overlap_cnv = chrom_cnv_df[
                        (chrom_cnv_df['Start'] < bin_global_end) &
                        (chrom_cnv_df['End'] > bin_global_start)
                    ]
                    if not overlap_cnv.empty:
                        cnv_type = overlap_cnv.iloc[0]['Type']
                        if cnv_type == 'DEL':
                            labels[bin_idx] = 1
                        elif cnv_type == 'DUP':
                            labels[bin_idx] = 2

                # 附加 bin_start 用于后续排序或分析
                sub_df['bin_start'] = [int_start + i*self.bin_size for i in range(N_bins)]

                all_sub_dfs.append(sub_df)
                all_sub_labels.append(labels)
                
                self.logger.info(f"[{chrom}] 区间 {idx+1}/{len(final_intervals)} 处理完成，生成了 {N_bins} 个bin，用时 {time.time() - interval_start_time:.2f} 秒")
            
            except Exception as e:
                self.logger.error(f"[{chrom}] 处理区间 {int_start}-{int_end} 失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue

        if not all_sub_dfs:
            self.logger.info(f"[{chrom}] 0 bins in final intervals, skipping.")
            return None

        features_df = pd.concat(all_sub_dfs, ignore_index=True)
        labels_all = np.concatenate(all_sub_labels)
        
        self.logger.info(
            f"[{chrom}] process_chromosome done. Bins: {len(features_df)}, near/far/total intervals: {len(merged)}/{len(far_regions)}/{len(final_intervals)}, 总用时 {time.time() - start_time:.2f} 秒"
        )

        return (features_df, labels_all)

    def build_dataset(self, bam_file, meth_file, cnv_bed_file, reference_file, chrom_lengths_dict=None):
        """
        构建训练集，提取指定区段内的特征并贴标签。
        Parameters:
        -----------
        bam_file: str
            BAM 文件路径
        meth_file: str
            甲基化数据文件路径
        cnv_bed_file: str
            CNV 标记 BED 文件路径
        reference_file: str
            参考基因组 FASTA 文件路径
        chrom_lengths_dict: dict
            染色体长度字典，可选
        
        Returns:
        --------
        X: pd.DataFrame
            特征矩阵
        y: np.ndarray
            标签向量
        """
        # 读取 CNV bed 文件
        try:
            cnv_df = pd.read_csv(cnv_bed_file, sep='\t')
            cnv_df['Chr'] = cnv_df['Chr'].str.replace('chr', '')
            cnv_df['Start'] = pd.to_numeric(cnv_df['Start'])
            cnv_df['End'] = pd.to_numeric(cnv_df['End'])
            
            
            self.logger.info(f"读取到 {len(cnv_df)} 个CNV记录")
        except Exception as e:
            self.logger.error(f"读取CNV文件失败: {str(e)}")
            raise

        # 根据线程数创建进程池
        if self.n_processes is None:
            self.n_processes = multiprocessing.cpu_count() - 2
        
        self.logger.info(f"使用 {self.n_processes} 个并行进程")
        process_pool = multiprocessing.Pool(self.n_processes)
        
        # 按染色体并行处理
        all_chromosomes = sorted(cnv_df['Chr'].unique())
        self.logger.info(f"发现 {len(all_chromosomes)} 条染色体: {all_chromosomes}")
        
        chromosomes = []
        pool_rets = []
        
        for chrom in all_chromosomes:
            chrom_cnv_df = cnv_df[cnv_df['Chr'] == chrom]
            
            # 如果没有 CNV 记录，跳过该染色体
            if len(chrom_cnv_df) == 0:
                self.logger.info(f"染色体 {chrom} 没有CNV记录，跳过")
                continue
                
            chromosomes.append(chrom)
            
            # 获取染色体长度
            chr_len = None
            if chrom_lengths_dict and chrom in chrom_lengths_dict:
                chr_len = chrom_lengths_dict[chrom]
            
            # 日志：开始处理该染色体
            self.logger.info(f"提交任务处理染色体 {chrom}, length={chr_len}")

            pool_rets.append(
                process_pool.apply_async(
                    self.process_chromosome,
                    (chrom, bam_file, meth_file, reference_file, chrom_cnv_df, chr_len)
                )
            )

        process_pool.close()
        
        # 收集结果
        all_features = []
        all_labels = []
        
        for chrom, ret in zip(chromosomes, pool_rets):
            self.logger.info(f"等待染色体 {chrom} 处理结果...")
            try:
                result = ret.get()
                if result is not None:
                    feat_df, labs = result
                    all_features.append(feat_df)
                    all_labels.append(labs)
                    self.logger.info(f"染色体 {chrom} 处理完成。结果中的bin数量: {len(feat_df)}")
                else:
                    self.logger.info(f"染色体 {chrom} 返回None结果。")
            except Exception as e:
                self.logger.error(f"处理染色体 {chrom} 时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        process_pool.join()

        if not all_features:
            raise ValueError("区域提取后没有有效数据")

        X = pd.concat(all_features, ignore_index=True)
        y = np.concatenate(all_labels)
        self.logger.info(f"最终数据集大小 => X:{X.shape}, y:{y.shape}")
        return X, y
# =========================================
# main 函数
# =========================================
def main():
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 如果没有手动提供 chrom_lengths_dict，就自动从 reference_file 获取
    reference_file = '/GLOBALFS/scau_xlyuan_3/zsh/DATA/BIYE_KETI/reference_genome/human/chr_num/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    chrom_lengths_dict = get_chrom_lengths_from_fasta(reference_file)
    print(chrom_lengths_dict)

    builder = TrainingDataBuilderRegion(
        bin_size=100,
        context_size=10000,
        normal_keep_fraction=0.0,
        n_processes=16
    )
    
    start_time = time.time()
    X, y = builder.build_dataset(
        bam_file='/GLOBALFS/scau_xlyuan_3/zsh/DATA/BIYE_KETI/WGBS_data/real_data/human/wgbs/bam/SRR20318450.bwameth.sorted.bam',
        meth_file='/GLOBALFS/scau_xlyuan_3/zsh/DATA/BIYE_KETI/Methylation_data/human/SRR20318450_CpG.bed',
        cnv_bed_file='/GLOBALFS/scau_xlyuan_3/zsh/DATA/BIYE_KETI/refCNV/human/reference_cnvs_human.bed',
        reference_file=reference_file,
        chrom_lengths_dict=chrom_lengths_dict
    )
    logging.info(f"数据集构建完成，总用时: {time.time() - start_time:.2f} 秒")
    
    # 添加对类别样本数量的统计
    class_counts = np.bincount(y)
    logging.info("=== 类别样本统计 ===")
    logging.info(f"标签 0 (Normal): {class_counts[0]} 个样本, 占比: {class_counts[0]/len(y):.2%}")
    logging.info(f"标签 1 (DEL): {class_counts[1]} 个样本, 占比: {class_counts[1]/len(y):.2%}")
    logging.info(f"标签 2 (DUP): {class_counts[2]} 个样本, 占比: {class_counts[2]/len(y):.2%}")
    logging.info("===================")
    
    # 保存特征和标签到NPZ文件
    np.savez('/GLOBALFS/scau_xlyuan_3/zsh/DATA/BIYE_KETI/MethylCNV_Penalty/buchong/data/train_data_human_bin100_rep2_14features.npz', 
             features=X, 
             labels=y,
             feature_names=X.columns)
    logging.info("数据已保存到train_data.npz")

if __name__ == "__main__":
    main() 