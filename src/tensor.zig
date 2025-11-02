const std: type = @import("std");
const Allocator: type = std.mem.Allocator;
const ArrayList: fn (type) type = std.ArrayList;

pub fn Tensor(T: type, shape_: []const usize) type {
    return struct {
        const Self = @This();
        pub const Index = [RANK]usize;

        elements: ArrayList(T), // use of the `T` parameter

        pub const RANK = SHAPE.len;
        pub const SHAPE = shape_;
        pub const number_of_elements = calculateNumberOfElements();
        pub fn rank(_: *const Self) usize {
            return RANK;
        }
        pub fn shape(_: *const Self) *const [RANK]usize {
            return SHAPE;
        }
        pub fn numberOfElements(_: *const Self) usize {
            return number_of_elements;
        }
        fn calculateNumberOfElements() usize {
            var totalLength: usize = 1;
            for (SHAPE) |dimensionLength| totalLength *= dimensionLength;
            return totalLength;
        }

        pub fn zeros(allocator: Allocator) Allocator.Error!Self {
            var elements = try ArrayList(T).initCapacity(allocator, Self.number_of_elements);
            for (0..number_of_elements) |_| try elements.append(allocator, 0);
            return .{ .elements = elements };
        }

        pub fn serializeIndex(index: *const Index) error{IndexOutOfBounds}!usize {
            var serialized_index: usize = 0;
            var stride: usize = 1;

            var dimension = RANK;
            while (dimension > 0) {
                dimension -= 1;
                if (index[dimension] >= SHAPE[dimension]) return error.IndexOutOfBounds;
                serialized_index += index[dimension] * stride;
                stride *= SHAPE[dimension];
            }

            return serialized_index;
        }
        pub fn deserializeIndex(index: usize) error{IndexOutOfBounds}![RANK]usize {
            var deserialized_index: Index = undefined;
            var remaining = index;

            var dimension = RANK;
            while (dimension > 0) {
                dimension -= 1;
                deserialized_index[dimension] = remaining % SHAPE[dimension];
                remaining /= SHAPE[dimension];
            }

            if (remaining != 0) return error.IndexOutOfBounds;

            return deserialized_index;
        }

        pub fn get(self: *Self, index: *const Index) error{IndexOutOfBounds}!*T {
            return &self.elements.items[try serializeIndex(index)];
        }
        pub fn set(self: *Self, index: *const Index, value: T) error{IndexOutOfBounds}!void {
            self.elements.items[try serializeIndex(index)] = value;
        }
    };
}

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;

test "serialize index" {
    const ExampleTensor = Tensor(f32, &.{ 7, 1, 12 });
    var expected_serialized_index: usize = 0;
    for (0..ExampleTensor.SHAPE[0]) |i| {
        for (0..ExampleTensor.SHAPE[1]) |j| {
            for (0..ExampleTensor.SHAPE[2]) |k| {
                const actual_serialized_index = try ExampleTensor.serializeIndex(&.{ i, j, k });
                try expect(actual_serialized_index == expected_serialized_index);
                expected_serialized_index += 1;
            }
        }
    }
}

test "deserialize index" {
    const ExampleTensor = Tensor(f32, &.{ 3, 3, 3 });
    var serial_index: usize = 0;
    for (0..ExampleTensor.SHAPE[0]) |i| {
        for (0..ExampleTensor.SHAPE[1]) |j| {
            for (0..ExampleTensor.SHAPE[2]) |k| {
                const actual_deserialized_index = try ExampleTensor.deserializeIndex(serial_index);
                const expected_deserialized_index = [ExampleTensor.RANK]usize{ i, j, k };
                try expect(std.mem.eql(usize, &actual_deserialized_index, &expected_deserialized_index));
                serial_index += 1;
            }
        }
    }
}
