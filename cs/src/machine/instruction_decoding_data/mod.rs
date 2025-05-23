use crate::devices::risc_v_types::NUM_INSTRUCTION_TYPES;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecoderMajorInstructionFamilyKey(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecoderInstructionVariantsKey(pub &'static str);

#[derive(Clone, Debug)]
pub struct DecoderOutputExtraKeysHolder {
    pub(crate) map:
        BTreeMap<DecoderMajorInstructionFamilyKey, BTreeSet<DecoderInstructionVariantsKey>>,
}

impl DecoderOutputExtraKeysHolder {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn num_major_keys(&self) -> usize {
        self.map.len()
    }

    pub fn all_major_keys(&self) -> Vec<DecoderMajorInstructionFamilyKey> {
        self.map.keys().cloned().collect()
    }

    pub fn max_minor_keys(&self) -> usize {
        let mut max = 0;
        for (_, v) in self.map.iter() {
            max = std::cmp::max(max, v.len());
        }

        max
    }

    pub fn collect(
        &mut self,
        major: DecoderMajorInstructionFamilyKey,
        minors: &[DecoderInstructionVariantsKey],
    ) {
        let entry = self.map.entry(major).or_default();
        for minor in minors.iter() {
            let _ = entry.insert(minor.clone());
        }
        // let is_unique = entry.insert(minor);
        // assert!(is_unique, "trying to add minor key {:?} again for major key {:?}", minor, major);
    }

    pub fn get_major_index(&self, major: &DecoderMajorInstructionFamilyKey) -> usize {
        let major_index = self.map.iter().position(|(k, _)| k == major).unwrap();

        major_index
    }

    #[track_caller]
    pub fn get_index_set(
        &self,
        major: &DecoderMajorInstructionFamilyKey,
        minor: &DecoderInstructionVariantsKey,
    ) -> (usize, usize) {
        let major_index = self.map.iter().position(|(k, _)| k == major).unwrap();
        let minors = &self.map.get(major).unwrap();
        let minor_index = minors.iter().position(|k| k == minor).unwrap();

        (major_index, minor_index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionOperandSelectionData {
    pub is_invalid: bool,
    pub src1_is_reg: bool,
    pub src2_is_reg: bool,
    pub src1_is_imm: bool,
    pub rs1_index_as_imm: bool,
    pub writes_rd: bool,
}

impl InstructionOperandSelectionData {
    pub fn as_integer(&self) -> u64 {
        (self.is_invalid as u64) << InstructionOperandSelectionDataProperties::IsInvalid as usize
            | (self.src1_is_reg as u64)
                << InstructionOperandSelectionDataProperties::UseRegForSrc1 as usize
            | (self.src2_is_reg as u64)
                << InstructionOperandSelectionDataProperties::UseRegForSrc2 as usize
            | (self.src1_is_imm as u64)
                << InstructionOperandSelectionDataProperties::UseImmForSrc1 as usize
            | (self.rs1_index_as_imm as u64)
                << InstructionOperandSelectionDataProperties::Rs1AsImm as usize
            | (self.writes_rd as u64)
                << InstructionOperandSelectionDataProperties::WriteToRd as usize
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionOperandSelectionDataProperties {
    IsInvalid = 0,
    UseRegForSrc1,
    UseRegForSrc2,
    UseImmForSrc1,
    Rs1AsImm,
    WriteToRd,
}

pub const NUM_INSTRUCTION_OPERAND_SELECTION_PROPS: usize =
    const { InstructionOperandSelectionDataProperties::WriteToRd as u32 as usize + 1 };

// R-type opcodes do not have immediate, so we do not need to perform extra work there eventually
pub const NUM_INSTRUCTION_TYPES_IN_DECODE_BITS: usize = NUM_INSTRUCTION_TYPES;

pub const BASE_INVALID_INSTRUCTION_DATA: InstructionOperandSelectionData =
    InstructionOperandSelectionData {
        is_invalid: true,
        src1_is_reg: false,
        src2_is_reg: false,
        src1_is_imm: false,
        rs1_index_as_imm: false,
        writes_rd: false,
    };

pub const BASE_R_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: true,
    src2_is_reg: true,
    src1_is_imm: false,
    rs1_index_as_imm: false,
    writes_rd: true,
};

pub const BASE_U_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: false,
    src2_is_reg: false,
    src1_is_imm: true,
    rs1_index_as_imm: false,
    writes_rd: true,
};

pub const BASE_J_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: false,
    src2_is_reg: false,
    src1_is_imm: false,
    rs1_index_as_imm: false,
    writes_rd: true,
};

pub const BASE_B_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: true,
    src2_is_reg: true,
    src1_is_imm: false,
    rs1_index_as_imm: false,
    writes_rd: false,
};

pub const BASE_S_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: true,
    src2_is_reg: true,
    src1_is_imm: false,
    rs1_index_as_imm: false,
    writes_rd: false,
};

pub const BASE_I_TYPE_AUX_DATA: InstructionOperandSelectionData = InstructionOperandSelectionData {
    is_invalid: false,
    src1_is_reg: true,
    src2_is_reg: false,
    src1_is_imm: false,
    rs1_index_as_imm: false,
    writes_rd: true,
};

pub const BASE_CSR_TYPE_AUX_DATA: InstructionOperandSelectionData =
    InstructionOperandSelectionData {
        is_invalid: false,
        src1_is_reg: true,
        src2_is_reg: false,
        src1_is_imm: false,
        rs1_index_as_imm: false,
        writes_rd: true,
    };

pub const BASE_CSRI_TYPE_AUX_DATA: InstructionOperandSelectionData =
    InstructionOperandSelectionData {
        is_invalid: false,
        src1_is_reg: false,
        src2_is_reg: false,
        src1_is_imm: false,
        rs1_index_as_imm: true,
        writes_rd: true,
    };
