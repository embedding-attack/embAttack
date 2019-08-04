import enum


class FeatureType(enum.Enum):
    DIFF_BIN_WITH_DIM = "diff_bins_num_{}_and_norm_dim"
    DIFF_BIN_WITH_DIM_2_HOP = "diff_bins_num_{}_and_norm_dim_2_hop"
    QCUT_DIST = "qcut_dist_bins_{}"
    EVEN_DIST = "even_dist_bins_{}"
    LOG_DIST = "log_dist_bins_{}"
    QUADRATIC_DIST = "quadratic_dist_bins_{}"

    def to_str(self, num_bins: int):
        return self.value.format(num_bins)

    @staticmethod
    def generate_feature_types():
        yield FeatureType.DIFF_BIN_WITH_DIM
        yield FeatureType.QCUT_DIST
        yield FeatureType.EVEN_DIST
        yield FeatureType.LOG_DIST
        yield FeatureType.QUADRATIC_DIST

    @staticmethod
    def get_readable_feature_type_map():
        return {str(FeatureType.DIFF_BIN_WITH_DIM):"Standard",
                str(FeatureType.QCUT_DIST):'Reused Bins',
                str(FeatureType.EVEN_DIST):'Equal',
                str(FeatureType.LOG_DIST):'Log',
                str(FeatureType.QUADRATIC_DIST):'Quadratic'}