const tensor = @import("tensor");
const Tensor = tensor.Tensor;

const std = @import("std");
const DefaultGeneralPurposeAllocator = std.heap.GeneralPurposeAllocator(.{});

pub fn main() !void {
    var generalPurposeAllocator = DefaultGeneralPurposeAllocator.init;
    const allocator = generalPurposeAllocator.allocator();
    _ = allocator;
}
